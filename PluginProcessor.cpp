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
}

Generativedelay14AudioProcessor::~Generativedelay14AudioProcessor()
{
    shutdownMelodyService();
}

bool Generativedelay14AudioProcessor::launchMelodyService()
{
    shutdownMelodyService();
    if (!melodyServiceExe.existsAsFile())
    {
        DBG("melody_service.exe not found at: " + melodyServiceExe.getFullPathName());
        return false;
    }

    SECURITY_ATTRIBUTES saAttr = { sizeof(SECURITY_ATTRIBUTES), NULL, TRUE };
    if (!CreatePipe(&hChildStdoutRd, &hChildStdoutWr, &saAttr, 0) ||
        !SetHandleInformation(hChildStdoutRd, HANDLE_FLAG_INHERIT, 0))
    {
        DBG("Pipe setup failed: " + juce::String(GetLastError()));
        return false;
    }

    STARTUPINFO si = { sizeof(STARTUPINFO) };
    si.hStdOutput = hChildStdoutWr;
    si.hStdError = hChildStdoutWr;
    si.dwFlags |= STARTF_USESTDHANDLES;

    juce::String exePath = melodyServiceExe.getFullPathName();
    std::vector<char> exePathBuffer(exePath.toStdString().begin(), exePath.toStdString().end());
    exePathBuffer.push_back('\0');

    if (!CreateProcess(exePathBuffer.data(), NULL, NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi))
    {
        DBG("CreateProcess failed: " + juce::String(GetLastError()));
        CloseHandle(hChildStdoutRd);
        CloseHandle(hChildStdoutWr);
        return false;
    }

    CloseHandle(hChildStdoutWr);
    hChildStdoutWr = nullptr;

    outputReaderThread = std::make_unique<OutputReaderThread>(hChildStdoutRd, responseQueue, responseQueueMutex);
    outputReaderThread->startThread();
    return true;
}

void Generativedelay14AudioProcessor::shutdownMelodyService()
{
    DBG("Shutting down melody service...");

    // Stop the thread first
    if (outputReaderThread)
    {
        // Close the pipe to unblock ReadFile
        if (hChildStdoutRd)
        {
            CloseHandle(hChildStdoutRd);
            hChildStdoutRd = nullptr;
        }
        outputReaderThread->stopThread(2000); // Wait up to 2 seconds
        outputReaderThread.reset();
        DBG("OutputReaderThread stopped");
    }

    // Kill the process
    if (pi.hProcess)
    {
        TerminateProcess(pi.hProcess, 0);
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
        pi.hProcess = nullptr;
        pi.hThread = nullptr;
        DBG("melody_service.exe terminated");
    }

    DBG("Shutdown complete");
}

void Generativedelay14AudioProcessor::OutputReaderThread::run()
{
    char buffer[1024];
    DWORD bytesRead;

    while (!threadShouldExit())
    {
        if (ReadFile(hPipe, buffer, sizeof(buffer) - 1, &bytesRead, NULL))
        {
            buffer[bytesRead] = '\0';
            std::string output(buffer);
            {
                juce::ScopedLock lock(cs);
                responseQueue.push(output);
            }
            DBG("melody_service.exe: " + juce::String(output));
        }
        else
        {
            if (GetLastError() == ERROR_BROKEN_PIPE)
            {
                DBG("Pipe closed, exiting thread");
                break;
            }
            DBG("ReadFile failed: " + juce::String(GetLastError()));
            juce::Thread::sleep(10);
        }
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

    if (!launchMelodyService())
    {
        DBG("Could not start melody_service.exe. Melody generation will be disabled.");
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
        //if (!generatedMelody.empty())
        //{
        //    const std::string& currentSymbol = generatedMelody[playbackPosition % generatedMelody.size()];

        //    // Handle current note state first
        //    if (playbackNoteActive && currentSymbol != "-")
        //    {
        //        // Stop current note if we're not continuing it
        //        midiMessages.addEvent(juce::MidiMessage::noteOff(2, playbackNote), 0);
        //        playbackNoteActive = false;
        //    }

        //    // Process new symbol
        //    if (std::all_of(currentSymbol.begin(), currentSymbol.end(), ::isdigit))
        //    {
        //        // Start new note
        //        playbackNote = std::stoi(currentSymbol);
        //        midiMessages.addEvent(juce::MidiMessage::noteOn(2, playbackNote, (uint8_t)vel), 0);
        //        playbackNoteActive = true;
        //    }
        //    // Note: for "-" we do nothing (continues current note)
        //    // Note: for "_" we've already handled any needed note-off above
        //    playbackPosition++;
        //}

        if (!generatedMelody.empty())
        {
            const std::string& currentSymbol = generatedMelody[playbackPosition % generatedMelody.size()];

            if (playbackNoteActive && currentSymbol != "-")
            {
                midiMessages.addEvent(juce::MidiMessage::noteOff(2, playbackNote), 0);
                playbackNoteActive = false;
            }

            if (currentSymbol != "_" && currentSymbol != "-")
            {
                try
                {
                    playbackNote = std::stoi(currentSymbol);
                    midiMessages.addEvent(juce::MidiMessage::noteOn(2, playbackNote, (uint8_t)vel), 0);
                    playbackNoteActive = true;
                }
                catch (const std::exception& e)
                {
                    DBG("Invalid symbol: " + juce::String(currentSymbol) + " - " + e.what());
                }
            }
            playbackPosition++;
        }

        captureCounter -= samplesPerSymbol;
    }
    captureCounter += buffer.getNumSamples();




    // process responses from melody_service.exe
    std::string response;
    {
        juce::ScopedLock lock(responseQueueMutex);
        if (!responseQueue.empty())
        {
            response = responseQueue.front();
            responseQueue.pop();
        }
    }
    if (!response.empty())
    {
        DBG("melody_service.exe: " + juce::String(response));
        // If the response is a generated melody, parse it
        if (response.find(" ") != std::string::npos) // Basic check for melody format
        {
            std::istringstream iss(response);
            std::string token;
            generatedMelody.clear();
            while (iss >> token)
            {
                generatedMelody.push_back(token);
            }
            playbackPosition = 0; // Reset playback for new melody
        }
    }




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
    melodyStr = melodyStr.trimEnd();
    DBG("Input melody: " + melodyStr);

    // Construct the prompt with parameters
    juce::String prompt = melodyStr + "|" + juce::String(temp) + "|1000\n"; // Adding newline to signal end of input
    std::string promptStr = prompt.toStdString();

    // Send the prompt to melody_service.exe
    if (hChildStdinWr)
    {
        DWORD bytesWritten;
        if (!WriteFile(hChildStdinWr, promptStr.c_str(), promptStr.length(), &bytesWritten, NULL))
        {
            DBG("Failed to write to melody_service.exe: " + juce::String(GetLastError()));
        }
        else
        {
            DBG("Sent prompt to melody_service.exe: " + prompt);
        }
    }
    else
    {
        DBG("Cannot send prompt: melody_service.exe not running");
    }

    playbackPosition = 0;
    bottleCap = false;
}






juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new Generativedelay14AudioProcessor();
}
