#pragma once
#include <JuceHeader.h>

class PianoRoll : public juce::Component
{
public:
    PianoRoll()
    {
        setSize(600, 480); // Fixed size to match visible area
    }

    void paint(juce::Graphics& g) override
    {
        // Fill background
        g.fillAll(juce::Colour(56, 0, 61));

        drawGrid(g);

        // Draw melodies
        if (!melody1.empty())
            drawNotes(g, melody1, juce::Colour(203, 174, 13));
        if (!melody2.empty())
            drawNotes(g, melody2, juce::Colour(18, 214, 158));
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
        const float rowHeight = getHeight() / 128.0f;

        // Draw vertical lines
        g.setColour(juce::Colour(69, 18, 75));
        for (int i = 0; i <= 32; ++i)
        {
            float x = i * columnWidth;
            g.fillRect(x, 0.0f, 1.0f, (float)getHeight());
        }

        // Draw horizontal lines for MIDI notes
        for (int i = 0; i <= 128; ++i)
        {
            float y = i * rowHeight;
            g.fillRect(0.0f, y, (float)getWidth(), 1.0f);
        }
    }

    void drawNotes(juce::Graphics& g, const std::vector<std::string>& melody, juce::Colour color)
    {
        const float columnWidth = getWidth() / 32.0f;
        const float rowHeight = getHeight() / 128.0f;

        g.setColour(color);

        for (int i = 0; i < melody.size(); ++i)
        {
            const auto& symbol = melody[i];
            if (symbol == "-" || symbol == "_")
                continue;

            try {
                int noteNumber = std::stoi(symbol);
                float x = i * columnWidth;
                float y = (127 - noteNumber) * rowHeight; // Invert Y so higher notes are at top
                float width = columnWidth;

                // Check for note continuation
                int duration = 1;
                for (int j = i + 1; j < melody.size() && melody[j] == "-"; ++j)
                    duration++;

                width = columnWidth * duration;

                g.fillRoundedRectangle(x, y, width, rowHeight, 2.0f);
            }
            catch (...) {
                continue;
            }
        }
    }

    std::vector<std::string> melody1;
    std::vector<std::string> melody2;
};