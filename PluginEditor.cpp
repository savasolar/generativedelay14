#include "PluginProcessor.h"
#include "PluginEditor.h"

Generativedelay14AudioProcessorEditor::Generativedelay14AudioProcessorEditor (Generativedelay14AudioProcessor& p)
    : AudioProcessorEditor (&p), audioProcessor (p)
{

    auto customFont = getCustomFont().withHeight(22.0f);

    // entropy labels & knob
    entropyLabel.setText("entropy", juce::dontSendNotification);
    entropyLabel.setFont(customFont);
    entropyLabel.setJustificationType(juce::Justification::centred);
    entropyLabel.setColour(juce::Label::textColourId, juce::Colours::white);
    entropyLabel.setBounds(25, 12.5, 70, 25);
    addAndMakeVisible(entropyLabel);

    entropyKnob.setAccentColour(juce::Colour(255, 119, 0));
    entropyKnob.setRange(0.0, 2.0, 0.1);
    entropyKnob.setValue(audioProcessor.getTemp());
    entropyKnob.addListener(this);
    entropyKnob.setBounds(25, 37.5, 70, 70);
    addAndMakeVisible(entropyKnob);

    entropyValueLabel.setText(juce::String(entropyKnob.getValue(), 1), juce::dontSendNotification);
    entropyValueLabel.setFont(customFont);
    entropyValueLabel.setJustificationType(juce::Justification::centred);
    entropyValueLabel.setColour(juce::Label::textColourId, juce::Colours::white);
    entropyValueLabel.setBounds(25, 107.5, 70, 25);
    addAndMakeVisible(entropyValueLabel);

    // rate (bpm) knob setup
    rateLabel.setText("rate", juce::dontSendNotification);
    rateLabel.setFont(customFont);
    rateLabel.setJustificationType(juce::Justification::centred);
    rateLabel.setColour(juce::Label::textColourId, juce::Colours::white);
    rateLabel.setBounds(25, 140.83, 70, 25);
    addAndMakeVisible(rateLabel);

    rateKnob.setAccentColour(juce::Colour(0, 119, 255));
    rateKnob.setRange(1.0, 300.0, 1.0);
    rateKnob.setValue(audioProcessor.getBpm());
    rateKnob.addListener(this);
    rateKnob.setBounds(25, 165.83, 70, 70);
    addAndMakeVisible(rateKnob);

    rateValueLabel.setText(juce::String(rateKnob.getValue(), 0), juce::dontSendNotification);
    rateValueLabel.setFont(customFont);
    rateValueLabel.setJustificationType(juce::Justification::centred);
    rateValueLabel.setColour(juce::Label::textColourId, juce::Colours::white);
    rateValueLabel.setBounds(25, 235.83, 70, 25);
    addAndMakeVisible(rateValueLabel);

    // velocity knob setup
    velLabel.setText("velocity", juce::dontSendNotification);
    velLabel.setFont(customFont);
    velLabel.setJustificationType(juce::Justification::centred);
    velLabel.setColour(juce::Label::textColourId, juce::Colours::white);
    velLabel.setBounds(25, 269.17, 70, 25);
    addAndMakeVisible(velLabel);

    velKnob.setAccentColour(juce::Colour(0, 255, 77));
    velKnob.setRange(0.0, 127.0, 1.0);
    velKnob.setValue(audioProcessor.getVel());
    velKnob.addListener(this);
    velKnob.setBounds(25, 294.17, 70, 70);
    addAndMakeVisible(velKnob);

    velValueLabel.setText(juce::String(velKnob.getValue(), 0), juce::dontSendNotification);
    velValueLabel.setFont(customFont);
    velValueLabel.setJustificationType(juce::Justification::centred);
    velValueLabel.setColour(juce::Label::textColourId, juce::Colours::white);
    velValueLabel.setBounds(25, 364.17, 70, 25);
    addAndMakeVisible(velValueLabel);

    // octave knob setup
    octaveLabel.setText("octave", juce::dontSendNotification);
    octaveLabel.setFont(customFont);
    octaveLabel.setJustificationType(juce::Justification::centred);
    octaveLabel.setColour(juce::Label::textColourId, juce::Colours::white);
    octaveLabel.setBounds(25, 397.5, 70, 25);
    addAndMakeVisible(octaveLabel);

    octaveKnob.setAccentColour(juce::Colour(0, 255, 255));
    octaveKnob.setRange(-4.0, 4.0, 1.0);
    octaveKnob.setValue(audioProcessor.getOctave());
    octaveKnob.addListener(this);
    octaveKnob.setBounds(25, 422.5, 70, 70);
    addAndMakeVisible(octaveKnob);

    octaveValueLabel.setText(juce::String(octaveKnob.getValue(), 0), juce::dontSendNotification);
    octaveValueLabel.setFont(customFont);
    octaveValueLabel.setJustificationType(juce::Justification::centred);
    octaveValueLabel.setColour(juce::Label::textColourId, juce::Colours::white);
    octaveValueLabel.setBounds(25, 492.5, 70, 25);
    addAndMakeVisible(octaveValueLabel);

    // info label setup
    infoLabel.setText("generative delay", juce::dontSendNotification);
    infoLabel.setFont(customFont);
    infoLabel.setJustificationType(juce::Justification::centredLeft);
    infoLabel.setColour(juce::Label::textColourId, juce::Colours::white);
    infoLabel.setBounds(135, 12.5, 200, 25);
    addAndMakeVisible(infoLabel);

    // browse, open buttons
    browseButton.setBounds(620, 0, 50, 50);
    addAndMakeVisible(browseButton);
    browseButton.onClick = [this]() { selectPlugin(); };

    openButton.setBounds(670, 0, 50, 50);
    openButton.setEnabled(false);
    addAndMakeVisible(openButton);
    openButton.onClick = [this]() { openPluginWindow(); };

    // set up plugin changed callback
    audioProcessor.pluginChanged = [this]() { pluginChanged(); };

    

    setUpVisualizer();
    startTimerHz(30);
    setSize (720, 530);
}

Generativedelay14AudioProcessorEditor::~Generativedelay14AudioProcessorEditor()
{
    stopTimer();
    //pianoRollViewport.setLookAndFeel(nullptr);
}

void Generativedelay14AudioProcessorEditor::paint (juce::Graphics& g)
{
    //g.fillAll (getLookAndFeel().findColour (juce::ResizableWindow::backgroundColourId));

    // Draw background sections
    g.setColour(juce::Colour(121, 88, 125));
    g.fillRect(0, 0, 120, 530);
    g.setColour(juce::Colour(21, 0, 23));
    g.fillRect(120, 0, 600, 50);


    g.drawImageAt(backgroundImage, 120, 50);
}

void Generativedelay14AudioProcessorEditor::resized()
{
    
    
    
    
}

void Generativedelay14AudioProcessorEditor::sliderValueChanged(juce::Slider* slider)
{
    if (slider == &entropyKnob.getSlider())
    {
        audioProcessor.setTemp(entropyKnob.getValue());
        entropyValueLabel.setText(juce::String(entropyKnob.getValue(), 1), juce::dontSendNotification);
    }
    else if (slider == &rateKnob.getSlider())
    {
        audioProcessor.setBpm(static_cast<int>(rateKnob.getValue()));
        rateValueLabel.setText(juce::String(rateKnob.getValue(), 0), juce::dontSendNotification);
    }
    else if (slider == &velKnob.getSlider())
    {
        audioProcessor.setVel(static_cast<int>(velKnob.getValue()));
        velValueLabel.setText(juce::String(velKnob.getValue(), 0), juce::dontSendNotification);
    }
    else if (slider == &octaveKnob.getSlider())
    {
        audioProcessor.setOctave(static_cast<int>(octaveKnob.getValue()));
        octaveValueLabel.setText(juce::String(octaveKnob.getValue(), 0), juce::dontSendNotification);
    }
}


void Generativedelay14AudioProcessorEditor::setUpVisualizer()
{

    // Create and add piano roll component directly
    pianoRoll = std::make_unique<PianoRoll>();
    pianoRoll->setMelodies(audioProcessor.getCapturedMelody(), audioProcessor.getGeneratedMelody());
    pianoRoll->setBounds(120, 50, 600, 480);
    addAndMakeVisible(pianoRoll.get());
}


void Generativedelay14AudioProcessorEditor::selectPlugin()
{
    // close existing windows first
    activeWindows.clear();

    chooser = std::make_unique<juce::FileChooser>(
        "Select a plugin...",
        juce::File(),
        "*.vst3;*.component"
    );

    auto flags = juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles;

    chooser->launchAsync(flags, [this](const juce::FileChooser& fc)
        {
            auto file = fc.getResult();

            if (file.exists())
            {
                auto& formatManager = audioProcessor.pluginFormatManager;

                juce::OwnedArray<juce::PluginDescription> descriptions;

                for (auto format : formatManager.getFormats())
                {
                    if (format->fileMightContainThisPluginType(file.getFullPathName()))
                    {
                        format->findAllTypesForFile(descriptions, file.getFullPathName());

                        if (descriptions.size() > 0)
                        {
                            audioProcessor.setNewPlugin(*descriptions[0]);
                            return;
                        }
                    }
                }
            }
        });
}

void Generativedelay14AudioProcessorEditor::openPluginWindow()
{
    // close existing windows first
    activeWindows.clear();

    if (!audioProcessor.isPluginLoaded())
        return;

    auto editor = audioProcessor.createInnerEditor();
    if (editor == nullptr)
        return;

    class PluginWindow : public juce::DocumentWindow
    {
    public:
        PluginWindow(const juce::String& name, juce::Colour backgroundColour)
            : DocumentWindow(name,
                backgroundColour,
                juce::DocumentWindow::minimiseButton | juce::DocumentWindow::closeButton)
        {
            setUsingNativeTitleBar(true);
            setResizable(true, false);
        }

        void closeButtonPressed() override
        {
            if (onClosing)
                onClosing();
        }

        std::function<void()> onClosing;
    };

    auto* window = new PluginWindow(audioProcessor.getLoadedPluginName(), juce::Colours::darkgrey);
    window->setContentOwned(editor.release(), true);
    window->centreWithSize(window->getWidth(), window->getHeight());

    window->onClosing = [this, window]()
        {
            activeWindows.removeObject(window);
        };

    activeWindows.add(window);
    window->setVisible(true);
}

void Generativedelay14AudioProcessorEditor::pluginChanged()
{
    openButton.setEnabled(audioProcessor.isPluginLoaded());

    // Update the info label with the loaded plugin name
    if (audioProcessor.isPluginLoaded())
    {
        infoLabel.setText("loaded plugin: " + audioProcessor.getLoadedPluginName(),
            juce::dontSendNotification);
    }
    else
    {
        infoLabel.setText("generative delay", juce::dontSendNotification);
    }
}