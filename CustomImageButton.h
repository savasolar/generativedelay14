// CustomImageButton.h
#pragma once
#include <JuceHeader.h>

class CustomImageButton : public juce::Component
{
public:
    CustomImageButton(const char* imageData, size_t imageSize)
    {
        image = juce::ImageCache::getFromMemory(imageData, imageSize);
    }

    void paint(juce::Graphics& g) override
    {
        // Draw background with three states:
        // 1. Pressed down - brightest
        // 2. Mouse over - medium bright
        // 3. Normal - darkest
        if (isMouseButtonDown)
            g.setColour(juce::Colour(21, 0, 23).brighter(0.1f));
        else if (isMouseOver())
            g.setColour(juce::Colour(21, 0, 23).brighter(0.2f));
        else
            g.setColour(juce::Colour(21, 0, 23));

        g.fillRect(getLocalBounds());

        // Draw image centered
        if (image.isValid())
        {
            const int x = (getWidth() - image.getWidth()) / 2;
            const int y = (getHeight() - image.getHeight()) / 2;
            g.drawImageAt(image, x, y, false);
        }
    }

    void mouseEnter(const juce::MouseEvent&) override
    {
        repaint();
    }

    void mouseExit(const juce::MouseEvent&) override
    {
        repaint();
    }

    void mouseDown(const juce::MouseEvent&) override
    {
        isMouseButtonDown = true;
        repaint();
        if (onClick)
            onClick();
    }

    void mouseUp(const juce::MouseEvent&) override
    {
        isMouseButtonDown = false;
        repaint();
    }

    std::function<void()> onClick;

private:
    juce::Image image;
    bool isMouseButtonDown = false;
};