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
int Generativedelay14AudioProcessor::getNumPrograms() { return 1; }
int Generativedelay14AudioProcessor::getCurrentProgram() { return 0; }
void Generativedelay14AudioProcessor::setCurrentProgram (int index) { }
const juce::String Generativedelay14AudioProcessor::getProgramName (int index) { return {}; }
void Generativedelay14AudioProcessor::changeProgramName (int index, const juce::String& newName) { }

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

    // store input midi
    juce::MidiBuffer inputMidi;
    inputMidi.addEvents(midiMessages, 0, buffer.getNumSamples(), 0);
    midiMessages.clear();

    // process incoming midi

    for (const auto metadata : inputMidi)
    {
        handleMidiMessage(metadata.getMessage());
    }

    //update capturedMelody based on timing

    if (captureCounter >= samplesPerSymbol)
    {
        std::string currentSlot = "_";

        if (noteIsOn)
        {
            // Check if previous slot had a note number
            int prevPos = (currentPosition - 1 + 32) % 32;
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
                currentSlot = std::to_string(lastPlayedNote);
            }
        }

        // update capturedmelody at current position

        capturedMelody[currentPosition % 32] = currentSlot;
        
        
        //DBG(capturedMelody[0] + " " + capturedMelody[1] + " " + capturedMelody[2] + " " + capturedMelody[3] + " " + capturedMelody[4] + " " + capturedMelody[5] + " " + capturedMelody[6] + " " + capturedMelody[7] + " " + capturedMelody[8] + " " + capturedMelody[9] + " " + capturedMelody[10] + " " + capturedMelody[11] + " " + capturedMelody[12] + " " + capturedMelody[13] + " " + capturedMelody[14] + " " + capturedMelody[15] + " " + capturedMelody[16] + " " + capturedMelody[17] + " " + capturedMelody[18] + " " + capturedMelody[19] + " " + capturedMelody[20] + " " + capturedMelody[21] + " " + capturedMelody[22] + " " + capturedMelody[23] + " " + capturedMelody[24] + " " + capturedMelody[25] + " " + capturedMelody[26] + " " + capturedMelody[27] + " " + capturedMelody[28] + " " + capturedMelody[29] + " " + capturedMelody[30] + " " + capturedMelody[31]);


        currentPosition++;

        // upon cycling through the entire capturedMelody array:
        if ((currentPosition % 32) == 0)
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

        captureCounter -= samplesPerSymbol;
    }
    captureCounter += buffer.getNumSamples();


    // add back the input MIDI
    midiMessages.addEvents(inputMidi, 0, buffer.getNumSamples(), 0);


    //if (!generatedMelody.empty())
    //{
    //    while (playbackCounter >= samplesPerSymbol)
    //    {
    //        const std::string& currentSymbol = generatedMelody[playbackPosition % generatedMelody.size()];

    //        if (!currentSymbol.empty())
    //        {
    //            if (std::all_of(currentSymbol.begin(), currentSymbol.end(), ::isdigit))
    //            {
    //                if (currentNoteNumber >= 0)
    //                {
    //                    midiMessages.addEvent(juce::MidiMessage::noteOff(2, currentNoteNumber), 0);
    //                }
    //                currentNoteNumber = std::stoi(currentSymbol);
    //                midiMessages.addEvent(juce::MidiMessage::noteOn(2, currentNoteNumber, (juce::uint8)100), 0);
    //            }
    //            else if (currentSymbol == "_" && currentNoteNumber >= 0)
    //            {
    //                midiMessages.addEvent(juce::MidiMessage::noteOff(2, currentNoteNumber), 0);
    //                currentNoteNumber = -1;
    //            }
    //        }

    //        playbackPosition++;
    //        playbackCounter -= samplesPerSymbol;
    //    }
    //    playbackCounter += buffer.getNumSamples();
    //}

    // Process through hosted plugin
    if (innerPlugin != nullptr)
    {
        innerPlugin->processBlock(buffer, midiMessages);
    }
    else
    {
        buffer.clear();
    }

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



void Generativedelay14AudioProcessor::handleMidiMessage(const juce::MidiMessage& message)
{
    if (message.isNoteOn())
    {
        currentMidiNote = message.getNoteNumber();
        lastPlayedNote = currentMidiNote;
        noteIsOn = true;
    }
    else if (message.isNoteOff() && message.getNoteNumber() == currentMidiNote)
    {
        noteIsOn = false;
        currentMidiNote = -1;
    }
}


void Generativedelay14AudioProcessor::generateNewMelody()
{
    // convert vector to string
    juce::String melodyString;
    for (const auto& note : capturedMelody) {
        melodyString += juce::String(note) + " ";
    }
    melodyString = melodyString.trimEnd();

    // construct json body
    juce::String jsonBody = "{\"prompt\": \"" + melodyString + "\"}";

    // launch network request on background thread ...
    std::thread([this, jsonBody]() {
        DBG(/*"Sending to Flask: " + */jsonBody);

        juce::URL url(flaskURL);
        juce::String response = url.withPOSTData(jsonBody).readEntireTextStream(true);

        if (response.isNotEmpty()) {
            // Process response and update UI on message thread
            juce::MessageManager::callAsync([this, response]() {
                DBG(/*"Received from Flask: " + */response);

                juce::var responseJson = juce::JSON::parse(response);
                if (responseJson.isObject()) {
                    juce::var generatedMelodyVar = responseJson["generated_melody"];
                    if (generatedMelodyVar.isString()) {
                        juce::String melodyString = generatedMelodyVar.toString();

                        // Parse response into vector
                        std::vector<std::string> newMelody;
                        std::istringstream iss(melodyString.toStdString());
                        std::string token;
                        while (std::getline(iss, token, ' ')) {
                            newMelody.push_back(token);
                        }

                        // Update generatedMelody
                        generatedMelody = std::move(newMelody);
                        //visuaLog("Updated generatedMelody, length: " + juce::String(generatedMelody.size()));


                        // indicate readiness for new capturedMelody
                        bottleCap = false;
                    }
                }
                });
        }
        else {
            juce::MessageManager::callAsync([this]() {
                DBG("Failed to connect to Flask server");
            });
        }
        }).detach();  // detach thread so it runs independently

}


juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new Generativedelay14AudioProcessor();
}
