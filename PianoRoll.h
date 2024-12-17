#pragma once
#include <JuceHeader.h>

class PianoRoll : public juce::Component
{
public:
    PianoRoll()
    {
        setSize(580, 3200); // 600 - 20 scrollbar width
    }

    void paint(juce::Graphics& g) override
    {
        // Fill background
        g.fillAll(juce::Colour(56, 0, 61));

        drawGrid(g);

        // Draw both melodies with new colors
        if (!melody1.empty())
        {
            drawNotes(g, melody1, juce::Colour(203, 174, 13).withAlpha(1.0f));  // Blue
        }
        if (!melody2.empty())
        {
            drawNotes(g, melody2, juce::Colour(18, 214, 158).withAlpha(1.0f));   // Green
        }
    }

    void setMelodies(const std::vector<std::string>& newMelody1, const std::vector<std::string>& newMelody2)
    {
        melody1 = newMelody1;
        melody2 = newMelody2;
        repaint();
    }

private:
    void drawGrid(juce::Graphics& g)
    {
        const float columnWidth = getWidth() / 32.0f;

        // Draw vertical lines
        g.setColour(juce::Colour(69, 18, 75));  // Slightly lighter purple for grid
        for (int i = 0; i <= 32; ++i)
        {
            float x = i * columnWidth;
            g.fillRect(x, 0.0f, 2.0f, (float)getHeight());
        }

        // Draw horizontal lines for notes
        const int numLines = getHeight() / noteHeight;
        for (int i = 0; i <= numLines; ++i)
        {
            float y = i * noteHeight;
            g.fillRect(0.0f, y, (float)getWidth(), 1.0f);
        }
    }

    void drawNotes(juce::Graphics& g, const std::vector<std::string>& melody, juce::Colour color)
    {
        const float columnWidth = getWidth() / 32.0f;
        const int centerY = getHeight() / 2;

        g.setColour(color);

        for (int i = 0; i < melody.size(); ++i)
        {
            const auto& symbol = melody[i];

            // Skip if it's a continuation or rest
            if (symbol == "-" || symbol == "_")
                continue;

            // Try to parse the note number
            try {
                int noteNumber = std::stoi(symbol);

                // Calculate note position and dimensions
                float x = i * columnWidth;
                float y = centerY + (centerNote - noteNumber) * noteHeight;
                float width = columnWidth;

                // Check for note continuation
                int duration = 1;
                for (int j = i + 1; j < melody.size() && melody[j] == "-"; ++j)
                    duration++;

                width = columnWidth * duration;

                // Draw the note rectangle with rounded corners
                g.fillRoundedRectangle(x, y, width, noteHeight, 4.0f);

                // Draw the note number
                g.setColour(juce::Colours::black.withAlpha(0.5f));
                g.drawText(symbol, juce::Rectangle<float>(x, y, width, noteHeight),
                    juce::Justification::centred, true);
                g.setColour(color);
            }
            catch (...) {
                continue;
            }
        }
    }

    const int noteHeight = 5;
    const int centerNote = 72;

    std::vector<std::string> melody1;
    std::vector<std::string> melody2;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PianoRoll)
};