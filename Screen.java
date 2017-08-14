package neuralNetwork;

import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.GridLayout;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;

import java.util.List;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;

import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix2D;

class DrawListender extends MouseAdapter {
    Pixel pixel;
    Screen screen;
    
    /**
     * Constructor.
     * 
     * @param pi The pixel that this listener is listening to.
     * */
    public DrawListender(Pixel pi, Screen scr) {
        pixel = pi;
        screen = scr;
    }
    
    /**
     * Triggers when the mouse is dragged.  draws a point on the pixel.
     * 
     * @param e The mouse Event.
     * */
    public void mouseEntered(MouseEvent ent) {
        if (SwingUtilities.isLeftMouseButton(ent))
            pixel.setIntensity(1);
        else if (SwingUtilities.isRightMouseButton(ent))
            pixel.setIntensity(0);
    }
}

public class Screen extends JFrame{
    static final int WIDTH  = 28,
                     HEIGHT = 28;
    Pixel[][] pixels = new Pixel[WIDTH][HEIGHT]; 
    Network network;
    
    public Screen(Network n) {
        network = n;
    }
    
    public void init()
    {
        setTitle("Draw a number!");
        setLayout(new FlowLayout());
        
        add(setupButtons());
        
        JPanel drawing = new JPanel();
        
        drawing.setLayout(new GridLayout(28, 28));

        for(int row = 0; row < HEIGHT; row++)
        {
            for(int col = 0; col < WIDTH; col++)
            {
                pixels[row][col] = new Pixel();
                pixels[row][col].addMouseListener(new DrawListender(pixels[row][col], this));
                drawing.add(pixels[row][col]);
            }
        }
        
        add(drawing);
    }
    
    private JPanel setupButtons() {
        //setup feedforward button
        JPanel buttons = new JPanel();
        
        JButton enterButton = new JButton();
        enterButton.setText("Enter");
        enterButton.addMouseListener(new MouseAdapter() {
            public void mouseClicked(MouseEvent ent) {
                blur();
                
                DoubleMatrix2D number = getImage();
                DoubleMatrix2D result = network.feedForward(number);
                
                for (int y = 0; y < 28; ++y){
                    for (int x = 0; x < 28; ++x) {
                        System.out.print((int)(number.get(y * 28 + x, 0) * 256) + "\t");
                    }
                    System.out.println();
               }
                
               System.out.println(result);
               System.out.println("Guess: " + Network.getMaxIndex(result));    
               
           	JOptionPane.showMessageDialog(null, "My guess is you wrote a\n" + Network.getMaxIndex(result), "Neural Network", 1);

           	reset();
            }            
        });
        
        JButton resetButton = new JButton();
        resetButton.setText("Reset");
        resetButton.addMouseListener(new MouseAdapter() {
            public void mouseClicked(MouseEvent ent) {
                reset();           
            }
        });
        
        JButton learnButton = new JButton();
        learnButton.setText("Learn");
        learnButton.addMouseListener(new MouseAdapter() {
            public void mouseClicked(MouseEvent ent) {
                JFrame jf = new JFrame("Input Dialog");           	
            	String userInput = JOptionPane.showInputDialog(jf, "Please enter number of epoch");
            	
            	int input = Integer.parseInt(userInput);
            	
                List<Image> training_data = Network.getImages(true);
                List<Image> test_data = Network.getImages(false); 
                
                network.SGD(training_data, input, 10, 3.0, test_data);
                
                JOptionPane.showMessageDialog(null, "I have finished learning!\nI am at " + String.format("%.2f",network.result) + "% accuracy now.", "Neural Network", 1);

                enterButton.setEnabled(true);
                resetButton.setEnabled(true);                
            }
        });
        
        buttons.add(enterButton);
        buttons.add(learnButton);
        buttons.add(resetButton);
        
        enterButton.setEnabled(false);
        resetButton.setEnabled(false);
        
        return buttons;
    }
    
    public void sizeScreen(int width, int height) {
        final Dimension size;

        size = new Dimension(width, height);
        setSize(size);
        
    }
    
    public void reset() {
        for(int row = 0; row < HEIGHT; row++)
            for(int col = 0; col < WIDTH; col++)
                pixels[row][col].setIntensity(0);
    }
    
    public void blur() {
        double[][] kernel = {
                {1.0/9.0, 1.0/9.0, 1.0/9.0},
                {1.0/9.0, 1.0/9.0, 1.0/9.0},
                {1.0/9.0, 1.0/9.0, 1.0/9.0}
        };
        
        for(int row = 1; row < HEIGHT - 1; row++) {
            for(int col = 1; col < WIDTH - 1; col++) {
                pixels[row][col].setIntensity(applyKernel(kernel, row, col));
            }
        }
    }
    
    public double applyKernel(double[][] a, int row, int col) {
        double sum = 0;
        
        for (int r = -1; r <= 1; ++r ) {
            for (int c = -1; c <= 1; ++c) {
                sum += a[r+1][c+1] * pixels[row + r][col + c].getIntensity();
            }
        }
        
        return sum;
    }
    
    public DoubleMatrix2D getImage() {
        DoubleMatrix2D image = DoubleFactory2D.dense.make(28 * 28, 1);
        int i = 0;
        for (int row = 0; row < 28; row++) {
            for (int column = 0; column < 28; column++) {
                image.set(i++, 0, pixels[row][column].getIntensity());
            }
        }
        
        return image;
    }
}
