#include "PluginProcessor.h"
#include "PluginEditor.h"
#include <thread>

Generativedelay14AudioProcessor::Generativedelay14AudioProcessor()
#ifndef JucePlugin_PreferredChannelConfigurations
     : AudioProcessor (BusesProperties()
                     #if ! JucePlugin_IsMidiEffect
                      #if ! JucePlugin_IsSynth
                       .withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
                      #endif
                       .withOutput ("Output", juce::AudioChannelSet::stereo(), true)
                     #endif
                       ),
    flaskURL("http://localhost:5000/generate_text")
#endif
{
    pluginFormatManager.addDefaultFormats();

    capturedMelody.resize(32, "_");
    generatedMelody.resize(32, "_");

    
}

Generativedelay14AudioProcessor::~Generativedelay14AudioProcessor()
{
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
//int Generativedelay14AudioProcessor::getNumPrograms() { return 1; }
//int Generativedelay14AudioProcessor::getCurrentProgram() { return 0; }
//void Generativedelay14AudioProcessor::setCurrentProgram (int index) { }
//const juce::String Generativedelay14AudioProcessor::getProgramName (int index) { return {}; }
//void Generativedelay14AudioProcessor::changeProgramName (int index, const juce::String& newName) { }

void Generativedelay14AudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    active = true;
    const juce::ScopedLock sl(innerMutex);


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
        
        //DBG(capturedMelody[0] + " " + capturedMelody[1] + " " + capturedMelody[2] + " " + capturedMelody[3] + " " + capturedMelody[4] + ...  + capturedMelody[31]);

        capturePosition++;

        // handle melody generation
        if ((capturePosition % 32) == 0)
        {
            // send to neural net for processing based on conditions

            // Check if capturedMelody contains any elements that aren't underscores
            bool hasNonUnderscores = std::any_of(capturedMelody.begin(), capturedMelody.end(),
                [](const std::string& element) { return element != "_"; });

            if (hasNonUnderscores)
            {
                bottleCap = false;
            }
            else
            {
                bottleCap = true;
            }


            if (bottleCap == false)
            {
                generateNewMelody();                
            }

            bottleCap = true; // prevent capturedMelody from sending more melodies to neural net before it is done generating

            std::fill(capturedMelody.begin(), capturedMelody.end(), "_");
        }


        // add playback functionality for generated melodies on a symbol-by-symbol basis here

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


    // add back the input MIDI
    midiMessages.addEvents(inputMidi, 0, buffer.getNumSamples(), 0);



    // Process through hosted plugin
    if (innerPlugin != nullptr)
    {
        innerPlugin->processBlock(buffer, midiMessages);
    }
    else
    {
        buffer.clear();
    }

    midiMessages.clear();
}

bool Generativedelay14AudioProcessor::hasEditor() const { return true; }

juce::AudioProcessorEditor* Generativedelay14AudioProcessor::createEditor()
{
    return new Generativedelay14AudioProcessorEditor (*this);
}

//==============================================================================
void Generativedelay14AudioProcessor::getStateInformation (juce::MemoryBlock& destData)
{

}

void Generativedelay14AudioProcessor::setStateInformation (const void* data, int sizeInBytes)
{

}


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
    


    // Initialize ML inference if not already done
    if (!mlInference)
    {

        DBG("Captured melody: (PluginProcessor)");

        juce::String melodyStr;
        for (const auto& token : capturedMelody) {
            melodyStr += juce::String(token) + " ";
        }
        DBG("Input melody: " + melodyStr.trimEnd());


        DBG("abt to load model");

        mlInference = std::make_unique<MelodicInference>();
        if (!mlInference->loadModel()) {
            DBG("Failed to load model");
            return;
        }

        juce::String resultStr;
        for (const auto& token : generatedMelody) {
            resultStr += juce::String(token) + " ";
        }
        DBG("Generated melody (PluginProcessor): " + resultStr.trimEnd());

    }


    // Convert capturedMelody directly
    generatedMelody = mlInference->generate(capturedMelody, temp, 200);

    // Reset playback
    playbackPosition = 0;
    bottleCap = false;
}


juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new Generativedelay14AudioProcessor();
}
