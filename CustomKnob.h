// CustomKnob.h
#pragma once
#include <JuceHeader.h>

class CustomKnob : public juce::Component
{
public:
    CustomKnob() : accentColour(juce::Colours::white) // default color
    {
        slider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
        slider.setTextBoxStyle(juce::Slider::NoTextBox, false, 0, 0);
        slider.setRange(0.0, 1.0);
        slider.setValue(0.0);
        addAndMakeVisible(slider);
        slider.setLookAndFeel(&customLookAndFeel);
    }

    ~CustomKnob()
    {
        slider.setLookAndFeel(nullptr);
    }

    // New functionality for range control
    void setRange(double newMinimum, double newMaximum, double newInterval = 0.0)
    {
        slider.setRange(newMinimum, newMaximum, newInterval);
    }

    // Value control
    void setValue(double newValue, juce::NotificationType notification = juce::sendNotificationAsync)
    {
        slider.setValue(newValue, notification);
    }

    double getValue() const
    {
        return slider.getValue();
    }

    // Listener functionality
    void addListener(juce::Slider::Listener* listener)
    {
        slider.addListener(listener);
    }

    void removeListener(juce::Slider::Listener* listener)
    {
        slider.removeListener(listener);
    }

    // Optional: Add getter for range information
    double getMinimum() const { return slider.getMinimum(); }
    double getMaximum() const { return slider.getMaximum(); }
    double getInterval() const { return slider.getInterval(); }

    void setAccentColour(juce::Colour colour)
    {
        accentColour = colour;
        repaint();
    }

    void resized() override
    {
        slider.setBounds(getLocalBounds());
    }

    juce::Slider& getSlider() { return slider; }

private:
    class PieChartLookAndFeel : public juce::LookAndFeel_V4
    {
    public:
        // In the PieChartLookAndFeel class within CustomKnob.h, replace the drawRotarySlider method:
        void drawRotarySlider(juce::Graphics& g, int x, int y, int width, int height,
            float sliderPos, float rotaryStartAngle, float rotaryEndAngle,
            juce::Slider& slider) override
        {
            // Get the accent colour from the parent CustomKnob
            auto* customKnob = static_cast<CustomKnob*>(slider.getParentComponent());
            auto accentColour = customKnob ? customKnob->accentColour : juce::Colours::white;

            auto bounds = juce::Rectangle<float>(x, y, width, height);
            auto radius = juce::jmin(bounds.getWidth(), bounds.getHeight()) / 2.0f;
            auto centre = bounds.getCentre();

            // Base layer (pie chart)
            const float baseSize = 70.0f;
            auto baseBounds = juce::Rectangle<float>(baseSize, baseSize).withCentre(centre);
            auto baseRadius = baseSize / 2.0f;

            // Fill the entire circle in base color first
            g.setColour(juce::Colour(185, 168, 187));
            g.fillEllipse(baseBounds);

            // Calculate the normalized value position based on the slider's range
            float normalizedValue = (slider.getValue() - slider.getMinimum()) /
                (slider.getMaximum() - slider.getMinimum());

            // For bipolar knobs (like octave), adjust the start angle to be at the middle
            bool isBipolar = slider.getMinimum() < 0 && slider.getMaximum() > 0;
            float zeroPosition = 0.0f;

            if (isBipolar) {
                zeroPosition = -slider.getMinimum() / (slider.getMaximum() - slider.getMinimum());

                // Draw the accent colored portion
                if (slider.getValue() != 0.0f) {
                    juce::Path pie;
                    float startPos = rotaryStartAngle + (zeroPosition * (rotaryEndAngle - rotaryStartAngle));
                    float endPos = rotaryStartAngle + (normalizedValue * (rotaryEndAngle - rotaryStartAngle));

                    pie.addPieSegment(baseBounds.getX(), baseBounds.getY(),
                        baseBounds.getWidth(), baseBounds.getHeight(),
                        std::min(startPos, endPos),
                        std::max(startPos, endPos),
                        0.0f);
                    g.setColour(accentColour);
                    g.fillPath(pie);
                }
            }
            else {
                // For regular knobs, draw from start if value > 0
                if (normalizedValue > 0.0f) {
                    juce::Path pie;
                    pie.addPieSegment(baseBounds.getX(), baseBounds.getY(),
                        baseBounds.getWidth(), baseBounds.getHeight(),
                        rotaryStartAngle,
                        rotaryStartAngle + normalizedValue * (rotaryEndAngle - rotaryStartAngle),
                        0.0f);
                    g.setColour(accentColour);
                    g.fillPath(pie);
                }
            }

            // Top layer circle (foreground)
            const float mainSize = 50.0f;
            auto mainBounds = juce::Rectangle<float>(mainSize, mainSize).withCentre(centre);

            // Draw drop shadow
            {
                juce::DropShadow shadow(juce::Colour(0, 0, 0).withAlpha(0.25f), 4, juce::Point<int>(0, 0));
                juce::Path circlePath;
                circlePath.addEllipse(mainBounds);
                shadow.drawForPath(g, circlePath);
            }

            // Draw main circle
            g.setColour(juce::Colour(249, 249, 249));
            g.fillEllipse(mainBounds);

            // Draw indicator dot
            const float dotSize = 10.0f;
            const float dotAngle = rotaryStartAngle + normalizedValue * (rotaryEndAngle - rotaryStartAngle)
                - juce::MathConstants<float>::halfPi;
            const float dotRadius = (mainSize / 2.0f) - (dotSize / 2.0f) - 1.0f;
            const float dotX = centre.x + std::cos(dotAngle) * dotRadius - (dotSize / 2.0f);
            const float dotY = centre.y + std::sin(dotAngle) * dotRadius - (dotSize / 2.0f);

            g.setColour(juce::Colours::black);
            g.fillEllipse(dotX, dotY, dotSize, dotSize);
        }
    };

    PieChartLookAndFeel customLookAndFeel;
    juce::Slider slider;
    juce::Colour accentColour;
    friend class PieChartLookAndFeel;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(CustomKnob)
};