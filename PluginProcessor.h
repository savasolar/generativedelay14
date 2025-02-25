#pragma once

#include <JuceHeader.h>
#include "MelodicInference.h"

class Generativedelay14AudioProcessor  : public juce::AudioProcessor
{
public:
    Generativedelay14AudioProcessor();
    ~Generativedelay14AudioProcessor() override;
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
   #ifndef JucePlugin_PreferredChannelConfigurations
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
   #endif
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;
    const juce::String getName() const override;
    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String& newName) override {}
    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    juce::AudioPluginFormatManager pluginFormatManager;
    std::unique_ptr<juce::AudioPluginInstance> innerPlugin;
    std::function<void()> pluginChanged;

    void setNewPlugin(const juce::PluginDescription& pd, const juce::MemoryBlock& mb = {});
    bool isPluginLoaded() const;
    juce::String getLoadedPluginName() const;
    std::unique_ptr<juce::AudioProcessorEditor> createInnerEditor() const;
    void clearPlugin();

    void setTemp(float newTemp) { temp = newTemp; }
    float getTemp() const { return temp; }

    void setBpm(int newBpm)
    {
        bpm = newBpm;
        if (active)
            samplesPerSymbol = (60.0 / bpm) * getSampleRate() / 4.0;
    }

    int getBpm() const { return bpm; }

    void setVel(int newVel) { vel = newVel; }
    int getVel() const { return vel; }

    void setOctave(int newOctave) { octave = newOctave; }
    int getOctave() const { return octave; }

    const std::vector<std::string>& getCapturedMelody() const { return capturedMelody; };
    const std::vector<std::string>& getGeneratedMelody() const { return generatedMelody; };

    void generateNewMelody();

private:
    // global/plugin state
    bool active = false;
    juce::CriticalSection innerMutex;

    // user params
    float temp = 0.8;
    int bpm = 100;
    int vel = 100;
    int octave = 0;
    
    // melody storage
    std::vector<std::string> capturedMelody;
    std::vector<std::string> generatedMelody;

    // input capture
    int inputNote = -1;                 // current input midi note (-1 if none)
    bool inputNoteActive = false;       // whether input note is playing
    int lastInputNote = -1;             // last input note number
    int capturePosition = 0;            // position in capture buffer
    double captureCounter = 0.0;        // sample counter for capture timing

    // melody playback
    int playbackNote = -1;              // current playback note (-1 if none)
    bool playbackNoteActive = false;    // whether playback note is playing 
    int playbackPosition = 0;           // position in generated melody
    //double playbackCounter = 0.0;       // sample counter for playback timing

    // timing
    double samplesPerSymbol = 0.0;      // samples per musical symbol

    // generation control
    bool bottleCap = true;
    juce::URL flaskURL;

    std::unique_ptr<MelodicInference> mlInference;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (Generativedelay14AudioProcessor)
};
