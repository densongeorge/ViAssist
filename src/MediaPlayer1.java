import javax.swing.JFrame;
import uk.co.caprica.vlcj.component.EmbeddedMediaPlayerComponent;
import uk.co.caprica.vlcj.runtime.RuntimeUtil;
import com.sun.jna.NativeLibrary;
import javax.swing.JPanel;

class MediaPlayer1  {
    //private JFrame ourFrame = new JFrame();

    
    private EmbeddedMediaPlayerComponent ourMediaPlayer;
    
    private String mediapath = "";
  
    public MediaPlayer1(String vlcpath, String mediapath ) {
        this.mediapath = mediapath;
        NativeLibrary.addSearchPath(RuntimeUtil.getLibVlcLibraryName(),vlcpath);
        ourMediaPlayer = new EmbeddedMediaPlayerComponent();
        ObjectFinder.vid1.setContentPane(ourMediaPlayer);
        ObjectFinder.vid1.setSize(500,500);
        ObjectFinder.vid1.setVisible(true);
        ObjectFinder.vid1.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
         //ourFrame.setContentPane(ourMediaPlayer);
        
        //ourFrame.setSize(1200,800);
        //ourFrame.setVisible(true);
        //ourFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        
        
        }
    public MediaPlayer1(String vlcpath, String mediapath, boolean f ) {
        this.mediapath = mediapath;
        NativeLibrary.addSearchPath(RuntimeUtil.getLibVlcLibraryName(),vlcpath);
        ourMediaPlayer = new EmbeddedMediaPlayerComponent();
        ObjectFinder.vid2.setContentPane(ourMediaPlayer);
        ObjectFinder.vid2.setSize(500,500);
        ObjectFinder.vid2.setVisible(true);
        ObjectFinder.vid2.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        
        
        }
    
    public void run()
    {
        ourMediaPlayer.getMediaPlayer().playMedia(mediapath);
    }
}
