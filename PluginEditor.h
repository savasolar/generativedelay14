#pragma once

#include <JuceHeader.h>
#include "PluginProcessor.h"
#include "CustomKnob.h"
#include "ImageButton.h"
#include "PianoRoll.h"

static const juce::Font getCustomFont()
{
    static auto typeface = juce::Typeface::createSystemTypefaceFor(BinaryData::Inter18_ttf,
        BinaryData::Inter18_ttfSize);
    return juce::Font(typeface);
}

class CustomLookAndFeel : public juce::LookAndFeel_V4
{
public:
    int getDefaultScrollbarWidth() override
    {
        return 20; // Set custom scrollbar width to 25px
    }

    void drawScrollbar(juce::Graphics& g,
        juce::ScrollBar& scrollbar,
        int x, int y,
        int width, int height,
        bool isScrollbarVertical,
        int thumbStartPosition,
        int thumbSize,
        bool isMouseOver,
        bool isMouseDown) override
    {
        g.setColour(juce::Colour(69, 18, 75));
        g.fillRect(x, y, width, height);

        if (thumbSize > 0)
        {
            g.setColour(juce::Colour(121, 88, 125));
            if (isScrollbarVertical)
                //g.fillRect(x, thumbStartPosition, width, thumbSize);
                g.fillRoundedRectangle(x + 4, thumbStartPosition, width - 8, thumbSize, 4.0f);
            else
                //g.fillRect(thumbStartPosition, y, thumbSize, height);
                g.fillRoundedRectangle(thumbStartPosition + 4, y, thumbSize - 8, height, 4.0f);
        }
    }
};

class Generativedelay14AudioProcessorEditor  : public juce::AudioProcessorEditor,
    public juce::Slider::Listener,
    public juce::Timer
{
public:
    Generativedelay14AudioProcessorEditor (Generativedelay14AudioProcessor&);
    ~Generativedelay14AudioProcessorEditor() override;

    void paint (juce::Graphics&) override;
    void resized() override;
    void sliderValueChanged(juce::Slider* slider) override;



    void setUpVisualizer();

    void timerCallback() override
    {
        // Update piano roll with latest melodies
        if (pianoRoll != nullptr)
        {
            pianoRoll->setMelodies(audioProcessor.getCapturedMelody(),
                audioProcessor.getGeneratedMelody());
            pianoRoll->repaint();
        }
    }

private:


    //CustomKnob testKnob;

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

    ImageButton browseButton{ BinaryData::browse_png, BinaryData::browse_pngSize };
    ImageButton openButton{ BinaryData::open_png, BinaryData::open_pngSize };


    juce::Viewport pianoRollViewport;
    std::unique_ptr<PianoRoll> pianoRoll;

    std::unique_ptr<CustomLookAndFeel> customLookAndFeel;

    Generativedelay14AudioProcessor& audioProcessor;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (Generativedelay14AudioProcessorEditor)
};
