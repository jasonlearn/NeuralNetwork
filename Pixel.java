package neuralNetwork;

import java.awt.Color;

import javax.swing.JPanel;

public class Pixel extends JPanel {
    public double intensity;
    
    public Pixel() {
        this.setBackground(new Color(255 - (int)(1 * 255),
                                     255 - (int)(1 * 255),
                                     255 - (int)(1 * 255)));
    }
    
    public void setIntensity(double x) {
        intensity = x;
        
        this.setBackground(new Color((int)(x * 255),
                                     (int)(x * 255),
                                     (int)(x * 255)));
        repaint();
    }
    
    public double getIntensity() {
        return intensity;
    }
}
