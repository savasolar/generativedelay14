#pragma once

#include <JuceHeader.h>
//#include <atomic>

class Generativedelay14AudioProcessor  : public juce::AudioProcessor
{
public:
    Generativedelay14AudioProcessor();
    ~Generativedelay14AudioProcessor() override;

    // Standard AudioProcessor overrides
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

    // Plugin hosting
    juce::AudioPluginFormatManager pluginFormatManager;
    std::unique_ptr<juce::AudioPluginInstance> innerPlugin;
    std::function<void()> pluginChanged;
    void setNewPlugin(const juce::PluginDescription& pd, const juce::MemoryBlock& mb = {});
    bool isPluginLoaded() const;
    juce::String getLoadedPluginName() const;
    std::unique_ptr<juce::AudioProcessorEditor> createInnerEditor() const;
    void clearPlugin();

    // Parameter setters and getters
    void setTemp(float newTemp) { temp = newTemp; }
    float getTemp() const { return temp; }
    void setBpm(int newBpm) { bpm = newBpm; updateSamplesPerSymbol(); }
    int getBpm() const { return bpm; }
    void setVel(int newVel) { vel = newVel; }
    int getVel() const { return vel; }
    void setOctave(int newOctave) { octave = newOctave; }
    int getOctave() const { return octave; }

    // Melody access
    const std::vector<std::string>& getCapturedMelody() const { return capturedMelody; };
    const std::vector<std::string>& getGeneratedMelody() const { return generatedMelody; };
    void generateNewMelody();

private:
    // State variables
    bool active = false;
    juce::CriticalSection innerMutex;

    // Parameters
    float temp = 0.8; // maybe use 0.8f ? test separately
    int bpm = 100;
    int vel = 100;
    int octave = 0;
    
    // Melody data
    std::vector<std::string> capturedMelody{ 32, "_" };
    std::vector<std::string> generatedMelody{ 32, "_" };

    // input capture
    int inputNote = -1;                 // current input midi note (-1 if none)
    bool inputNoteActive = false;       // whether input note is playing
    int lastInputNote = -1;             // last input note number
    int capturePosition = 0;            // position in capture buffer
    double captureCounter = 0.0;        // sample counter for capture timing

    // Playback
    int playbackNote = -1;              // current playback note (-1 if none)
    bool playbackNoteActive = false;    // whether playback note is playing 
    int playbackPosition = 0;           // position in generated melody

    // Timing
    double samplesPerSymbol = 0.0;      // samples per musical symbol
    void updateSamplesPerSymbol() { if (active) samplesPerSymbol = (60.0 / bpm) * getSampleRate() / 4.0; }

    // Generation control
    bool bottleCap = true;

    // Melody service
    juce::File melodyServiceExe;
    std::unique_ptr<juce::ChildProcess> melodyService;



    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (Generativedelay14AudioProcessor)
};
