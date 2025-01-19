#pragma once

#include <JuceHeader.h>
#include "PluginProcessor.h"
#include "CustomKnob.h"
#include "CustomImageButton.h"
#include "PianoRoll.h"

static const juce::Font getCustomFont()
{
    static auto typeface = juce::Typeface::createSystemTypefaceFor(BinaryData::Inter18_ttf,
        BinaryData::Inter18_ttfSize);
    return juce::Font(typeface);
}

class Generativedelay14AudioProcessorEditor : public juce::AudioProcessorEditor,
    public juce::Slider::Listener,
    public juce::Timer
{
public:
    Generativedelay14AudioProcessorEditor(Generativedelay14AudioProcessor&);
    ~Generativedelay14AudioProcessorEditor() override;

    void paint(juce::Graphics&) override;
    void resized() override;
    void sliderValueChanged(juce::Slider* slider) override;
    void setUpVisualizer();

    void timerCallback() override
    {
        if (pianoRoll != nullptr)
        {
            pianoRoll->setMelodies(audioProcessor.getCapturedMelody(),
                audioProcessor.getGeneratedMelody());
            pianoRoll->repaint();
        }
    }

private:
    void selectPlugin();
    void openPluginWindow();
    void pluginChanged();

    juce::OwnedArray<juce::DocumentWindow> activeWindows;
    std::unique_ptr<juce::FileChooser> chooser;
    juce::Image backgroundImage;

    juce::Label entropyLabel;
    CustomKnob entropyKnob;
    juce::Label entropyValueLabel;

    juce::Label rateLabel;
    CustomKnob rateKnob;
    juce::Label rateValueLabel;

    juce::Label velLabel;
    CustomKnob velKnob;
    juce::Label velValueLabel;

    juce::Label octaveLabel;
    CustomKnob octaveKnob;
    juce::Label octaveValueLabel;

    juce::Label infoLabel;

    CustomImageButton browseButton{ BinaryData::browse_png, BinaryData::browse_pngSize };
    CustomImageButton openButton{ BinaryData::open_png, BinaryData::open_pngSize };

    std::unique_ptr<PianoRoll> pianoRoll;

    Generativedelay14AudioProcessor& audioProcessor;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(Generativedelay14AudioProcessorEditor)
};