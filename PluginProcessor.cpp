#include "PluginProcessor.h"
#include "PluginEditor.h"
#include <algorithm>

Generativedelay14AudioProcessor::Generativedelay14AudioProcessor()
#ifndef JucePlugin_PreferredChannelConfigurations
     : AudioProcessor (BusesProperties()
                     #if ! JucePlugin_IsMidiEffect
                      #if ! JucePlugin_IsSynth
                       .withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
                      #endif
                       .withOutput ("Output", juce::AudioChannelSet::stereo(), true)
                     #endif
                       )
#endif
{
    pluginFormatManager.addDefaultFormats();

    juce::File pluginFile = juce::File::getSpecialLocation(juce::File::currentApplicationFile);
    melodyServiceExe = pluginFile.getSiblingFile("melody_service.exe");

    
    // Check if the melody service exists
    if (melodyServiceExe.existsAsFile())
    {
        DBG("Starting melody service from: " + melodyServiceExe.getFullPathName());

        // Create melody service process
        melodyService = std::make_unique<juce::ChildProcess>();

        // Start the process with redirected output
        bool started = melodyService->start(melodyServiceExe.getFullPathName(),
            juce::ChildProcess::wantStdOut);

        if (started)
            DBG("Melody service started successfully!");
        else
            DBG("Failed to start melody service!");
    }
}

Generativedelay14AudioProcessor::~Generativedelay14AudioProcessor()
{
    // Clean up melody service
    if (melodyService != nullptr && melodyService->isRunning())
    {
        DBG("Shutting down melody service...");
        melodyService->kill();

        // Optional: Wait briefly to ensure the process has terminated
        int waitCount = 0;
        while (melodyService->isRunning() && waitCount < 10)
        {
            juce::Thread::sleep(100);
            waitCount++;
        }

        if (melodyService->isRunning())
            DBG("WARNING: Melody service failed to terminate cleanly!");
        else
            DBG("Melody service terminated successfully.");
    }

}

const juce::String Generativedelay14AudioProcessor::getName() const { return JucePlugin_Name; }

bool Generativedelay14AudioProcessor::acceptsMidi() const
{
   #if JucePlugin_WantsMidiInput
    return true;
   #else
    return true;
   #endif
}

bool Generativedelay14AudioProcessor::producesMidi() const
{
   #if JucePlugin_ProducesMidiOutput
    return true;
   #else
    return true;
   #endif
}

bool Generativedelay14AudioProcessor::isMidiEffect() const
{
   #if JucePlugin_IsMidiEffect
    return true;
   #else
    return false;
   #endif
}

double Generativedelay14AudioProcessor::getTailLengthSeconds() const { return 0.0; }

void Generativedelay14AudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    active = true;
    const juce::ScopedLock sl(innerMutex);
    updateSamplesPerSymbol();
    if (innerPlugin != nullptr)
    {
        innerPlugin->setRateAndBufferSizeDetails(sampleRate, samplesPerBlock);
        innerPlugin->prepareToPlay(sampleRate, samplesPerBlock);
    }
}

void Generativedelay14AudioProcessor::releaseResources()
{
    const juce::ScopedLock sl(innerMutex);
    active = false;
    if (innerPlugin != nullptr)
        innerPlugin->releaseResources();
}

#ifndef JucePlugin_PreferredChannelConfigurations
bool Generativedelay14AudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
  #if JucePlugin_IsMidiEffect
    juce::ignoreUnused (layouts);
    return true;
  #else

    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono()
     && layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

   #if ! JucePlugin_IsSynth
    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;
   #endif

    return true;
  #endif
}
#endif

void Generativedelay14AudioProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals;
    const juce::ScopedLock sl(innerMutex);

    // store and input midi
    juce::MidiBuffer inputMidi;
    inputMidi.addEvents(midiMessages, 0, buffer.getNumSamples(), 0);
    midiMessages.clear();

    // process incoming midi
    for (const auto metadata : inputMidi)
    {
        const auto& msg = metadata.getMessage();
        if (msg.isNoteOn())
        {
            inputNote = msg.getNoteNumber();
            lastInputNote = inputNote;
            inputNoteActive = true;
        }
        else if (msg.isNoteOff() && msg.getNoteNumber() == inputNote)
        {
            inputNoteActive = false;
            inputNote = -1;
        }
    }

    // handle melody capture timing
    if (captureCounter >= samplesPerSymbol)
    {
        std::string currentSlot = "_";

        if (inputNoteActive)
        {
            // Check if previous slot had a note number
            int prevPos = (capturePosition - 1 + 32) % 32;
            bool prevWasNumber = std::all_of(capturedMelody[prevPos].begin(),
                capturedMelody[prevPos].end(),
                ::isdigit);
            bool prevWasContinuation = capturedMelody[prevPos] == "-";

            // If we're continuing the same note (either from a number or continuation symbol)
            if (prevWasNumber || prevWasContinuation)
            {
                currentSlot = "-";
            }
            else
            {
                // Only write the note number if we're starting a new note
                currentSlot = std::to_string(lastInputNote);
            }
        }

        // update capturedmelody at current position
        capturedMelody[capturePosition % 32] = currentSlot;
        capturePosition++;

        // handle melody generation
        if ((capturePosition % 32) == 0)
        {
            DBG("Triggering melody generation");
            // send to neural net for processing based on conditions

            // Check if capturedMelody contains any elements that aren't underscores
            bool hasNonUnderscores = std::any_of(capturedMelody.begin(), capturedMelody.end(),
                [](const std::string& element) { return element != "_"; });

            if (hasNonUnderscores)
                bottleCap = false;
            else
                bottleCap = true;

            if (bottleCap == false)
                generateNewMelody();                

            bottleCap = true; // prevent capturedMelody from sending more melodies to neural net before it is done generating

            std::fill(capturedMelody.begin(), capturedMelody.end(), "_");
        }

        // add playback functionality for generated melodies on a symbol-by-symbol basis
        if (!generatedMelody.empty())
        {

            const std::string& currentSymbol = generatedMelody[playbackPosition % generatedMelody.size()];

            // Handle current note state first
            if (playbackNoteActive && currentSymbol != "-")
            {
                // Stop current note if we're not continuing it
                midiMessages.addEvent(juce::MidiMessage::noteOff(2, playbackNote), 0);
                playbackNoteActive = false;
            }

            // Process new symbol
            if (std::all_of(currentSymbol.begin(), currentSymbol.end(), ::isdigit))
            {
                // Start new note
                playbackNote = std::stoi(currentSymbol);
                midiMessages.addEvent(juce::MidiMessage::noteOn(2, playbackNote, (uint8_t)vel), 0);
                playbackNoteActive = true;
            }
            // Note: for "-" we do nothing (continues current note)
            // Note: for "_" we've already handled any needed note-off above
            playbackPosition++;
        }

        captureCounter -= samplesPerSymbol;
    }
    captureCounter += buffer.getNumSamples();

    // async communication
    processServiceCommunication();

    // Update generated melody if ready
    





    // add back the input MIDI
    midiMessages.addEvents(inputMidi, 0, buffer.getNumSamples(), 0);

    // Process through hosted plugin
    if (innerPlugin != nullptr)
        innerPlugin->processBlock(buffer, midiMessages);
    else
        buffer.clear();

    midiMessages.clear();
}

bool Generativedelay14AudioProcessor::hasEditor() const { return true; }

juce::AudioProcessorEditor* Generativedelay14AudioProcessor::createEditor()
{
    return new Generativedelay14AudioProcessorEditor (*this);
}

//==============================================================================
void Generativedelay14AudioProcessor::getStateInformation (juce::MemoryBlock& destData) {}

void Generativedelay14AudioProcessor::setStateInformation (const void* data, int sizeInBytes) {}


void Generativedelay14AudioProcessor::setNewPlugin(const juce::PluginDescription& pd, const juce::MemoryBlock& mb)
{
    const juce::ScopedLock sl(innerMutex);

    // First, clear any existing plugin
    if (innerPlugin != nullptr)
    {
        innerPlugin->releaseResources();
        innerPlugin = nullptr;
    }

    auto callback = [this, mb](std::unique_ptr<juce::AudioPluginInstance> instance, const juce::String& error)
        {
            if (error.isNotEmpty())
                return;

            const juce::ScopedLock sl(innerMutex);
            innerPlugin = std::move(instance);

            if (!mb.isEmpty())
                innerPlugin->setStateInformation(mb.getData(), (int)mb.getSize());

            if (active)
            {
                innerPlugin->setRateAndBufferSizeDetails(getSampleRate(), getBlockSize());
                innerPlugin->prepareToPlay(getSampleRate(), getBlockSize());
            }

            juce::MessageManager::callAsync([this]()
                {
                    juce::NullCheckedInvocation::invoke(pluginChanged);
                });
        };

    pluginFormatManager.createPluginInstanceAsync(pd, getSampleRate(), getBlockSize(), callback);
}

bool Generativedelay14AudioProcessor::isPluginLoaded() const
{
    const juce::ScopedLock sl(innerMutex);
    return innerPlugin != nullptr;
}

juce::String Generativedelay14AudioProcessor::getLoadedPluginName() const
{
    const juce::ScopedLock sl(innerMutex);
    return innerPlugin != nullptr ? innerPlugin->getName() : juce::String("empty");
}

std::unique_ptr<juce::AudioProcessorEditor> Generativedelay14AudioProcessor::createInnerEditor() const
{
    const juce::ScopedLock sl(innerMutex);
    return innerPlugin != nullptr && innerPlugin->hasEditor() ?
        std::unique_ptr<juce::AudioProcessorEditor>(innerPlugin->createEditorIfNeeded()) : nullptr;
}

void Generativedelay14AudioProcessor::clearPlugin()
{
    const juce::ScopedLock sl(innerMutex);
    innerPlugin = nullptr;
    juce::NullCheckedInvocation::invoke(pluginChanged);
}


void Generativedelay14AudioProcessor::generateNewMelody()
{
    DBG("Captured melody: (PluginProcessor)");

    juce::String melodyStr;
    for (const auto& token : capturedMelody) {
        melodyStr += juce::String(token) + " ";
    }
    DBG("Input melody: " + melodyStr.trimEnd());

    




    playbackPosition = 0;
    bottleCap = false;
}

void Generativedelay14AudioProcessor::processServiceCommunication() {
    
    // Check if service is still running
    if (melodyService != nullptr && !melodyService->isRunning())
    {
        DBG("WARNING: Melody service has terminated unexpectedly. Attempting to restart...");

        // Attempt to restart the service
        if (melodyServiceExe.existsAsFile())
        {
            bool started = melodyService->start(melodyServiceExe.getFullPathName());
            if (started)
                DBG("Melody service restarted successfully!");
            else
                DBG("Failed to restart melody service!");
        }
    }

    // Read and display console output from the melody service
    if (melodyService != nullptr && melodyService->isRunning())
    {
        juce::String output = melodyService->readAllProcessOutput();
        if (output.isNotEmpty())
        {
            // Split by line breaks to get each line of output
            juce::StringArray lines = juce::StringArray::fromLines(output);
            for (const juce::String& line : lines)
            {
                if (line.isNotEmpty())
                    DBG("Melody Service: " + line);
            }
        }
    }
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new Generativedelay14AudioProcessor();
}
