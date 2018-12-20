
import com.sun.speech.freetts.Voice;
import com.sun.speech.freetts.VoiceManager;
import java.awt.Image;
import java.sql.*;
import java.util.Arrays;
import java.awt.event.KeyEvent;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.logging.Logger;
import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFileChooser;
import javax.swing.JOptionPane;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.opencv_nonfree;
import org.bytedeco.javacpp.opencv_core.IplImage;
import static org.bytedeco.javacpp.opencv_calib3d.*;
import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_flann.*;
import static org.bytedeco.javacpp.opencv_highgui.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_legacy.*;
import org.bytedeco.javacv.BaseChildSettings;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameGrabber;
import uk.co.caprica.vlcj.player.MediaPlayer;

public class ObjectFinder extends javax.swing.JFrame {

    static Connection c = null;
    static ResultSet rs = null;
    static Statement stmt = null;
    static int cnt;
    static int crop;
    static String mediapath;
    static String mediapath1;
    static boolean detect = true;
    private static JFileChooser ourFileSelector = new JFileChooser();
    private static final String VOICENAME = "kevin16";

    static {
        Loader.load(opencv_nonfree.class);
    }

    public ObjectFinder() {
        initComponents();
        sproc.setEnabled(false);
        dest1.setEditable(false);
        busNo1.setEditable(false);
        busNo2.setEditable(false);
        dest2.setEditable(false);
    }

    @SuppressWarnings("unchecked")
    public ObjectFinder(IplImage objectImage) {
        settings = new Settings();
        settings.objectImage = objectImage;
        setSettings(settings);
    }

    public ObjectFinder(Settings settings) {
        setSettings(settings);
    }

    public static class Settings extends BaseChildSettings {

        IplImage objectImage = null;
        CvSURFParams parameters = cvSURFParams(500, 1);
        double distanceThreshold = 0.6;
        int matchesMin = 4;
        double ransacReprojThreshold = 1.0;
        boolean useFLANN = false;

        public IplImage getObjectImage() {
            return objectImage;
        }

        public void setObjectImage(IplImage objectImage) {
            this.objectImage = objectImage;
        }

        public boolean isExtended() {
            return parameters.extended() != 0;
        }

        public void setExtended(boolean extended) {
            parameters.extended(extended ? 1 : 0);
        }

        public boolean isUpright() {
            return parameters.upright() != 0;
        }

        public void setUpright(boolean upright) {
            parameters.upright(upright ? 1 : 0);
        }

        public double getHessianThreshold() {
            return parameters.hessianThreshold();
        }

        public void setHessianThreshold(double hessianThreshold) {
            parameters.hessianThreshold(hessianThreshold);
        }

        public int getnOctaves() {
            return parameters.nOctaves();
        }

        public void setnOctaves(int nOctaves) {
            parameters.nOctaves(nOctaves);
        }

        public int getnOctaveLayers() {
            return parameters.nOctaveLayers();
        }

        public void setnOctaveLayers(int nOctaveLayers) {
            parameters.nOctaveLayers(nOctaveLayers);
        }

        public double getDistanceThreshold() {
            return distanceThreshold;
        }

        public void setDistanceThreshold(double distanceThreshold) {
            this.distanceThreshold = distanceThreshold;
        }

        public int getMatchesMin() {
            return matchesMin;
        }

        public void setMatchesMin(int matchesMin) {
            this.matchesMin = matchesMin;
        }

        public double getRansacReprojThreshold() {
            return ransacReprojThreshold;
        }

        public void setRansacReprojThreshold(double ransacReprojThreshold) {
            this.ransacReprojThreshold = ransacReprojThreshold;
        }

        public boolean isUseFLANN() {
            return useFLANN;
        }

        public void setUseFLANN(boolean useFLANN) {
            this.useFLANN = useFLANN;
        }
    }
    private Settings settings;

    public Settings getSettings() {
        return settings;
    }

    public void setSettings(Settings settings) {
        this.settings = settings;

        CvSeq keypoints = new CvSeq(null), descriptors = new CvSeq(null);
        cvClearMemStorage(storage);
        cvExtractSURF(settings.objectImage, null, keypoints, descriptors, storage, settings.parameters, 0);

        int total = descriptors.total();
        int size = descriptors.elem_size();
        objectKeypoints = new CvSURFPoint[total];
        objectDescriptors = new FloatBuffer[total];
        for (int i = 0; i < total; i++) {
            objectKeypoints[i] = new CvSURFPoint(cvGetSeqElem(keypoints, i));
            objectDescriptors[i] = cvGetSeqElem(descriptors, i).capacity(size).asByteBuffer().asFloatBuffer();
        }
        if (settings.useFLANN) {
            int length = objectDescriptors[0].capacity();
            objectMat = new Mat(total, length, CV_32FC1);
            imageMat = new Mat(total, length, CV_32FC1);
            indicesMat = new Mat(total, 2, CV_32SC1);
            distsMat = new Mat(total, 2, CV_32FC1);

            flannIndex = new Index();
            indexParams = new KDTreeIndexParams(4); // using 4 randomized kdtrees
            searchParams = new SearchParams(64, 0, true); // maximum number of leafs checked
        }
        pt1 = CvMat.create(1, total, CV_32F, 2);
        pt2 = CvMat.create(1, total, CV_32F, 2);
        mask = CvMat.create(1, total, CV_8U, 1);
        H = CvMat.create(3, 3);
        ptpairs = new ArrayList<Integer>(2 * objectDescriptors.length);
        logger.info(total + " object descriptors");
    }
    private static final Logger logger = Logger.getLogger(ObjectFinder.class.getName());
    private CvMemStorage storage = CvMemStorage.create();
    private CvMemStorage tempStorage = CvMemStorage.create();
    private CvSURFPoint[] objectKeypoints = null, imageKeypoints = null;
    private FloatBuffer[] objectDescriptors = null, imageDescriptors = null;
    private Mat objectMat, imageMat, indicesMat, distsMat;
    private Index flannIndex = null;
    private IndexParams indexParams = null;
    private SearchParams searchParams = null;
    private CvMat pt1 = null, pt2 = null, mask = null, H = null;
    public static ArrayList<Integer> ptpairs = null;

    public double[] find(IplImage image) {
        CvSeq keypoints = new CvSeq(null), descriptors = new CvSeq(null);
        cvClearMemStorage(tempStorage);
        cvExtractSURF(image, null, keypoints, descriptors, tempStorage, settings.parameters, 0);

        int total = descriptors.total();
        int size = descriptors.elem_size();
        imageKeypoints = new CvSURFPoint[total];
        imageDescriptors = new FloatBuffer[total];
        for (int i = 0; i < total; i++) {
            imageKeypoints[i] = new CvSURFPoint(cvGetSeqElem(keypoints, i));
            imageDescriptors[i] = cvGetSeqElem(descriptors, i).capacity(size).asByteBuffer().asFloatBuffer();
        }
        logger.info(total + " image descriptors");

        int w = settings.objectImage.width();
        int h = settings.objectImage.height();
        double[] srcCorners = {0, 0, w, 0, w, h, 0, h};
        double[] dstCorners = locatePlanarObject(objectKeypoints, objectDescriptors, imageKeypoints, imageDescriptors, srcCorners);
        return dstCorners;
    }

    private double compareSURFDescriptors(FloatBuffer d1, FloatBuffer d2, double best) {
        double totalCost = 0;
        assert (d1.capacity() == d2.capacity() && d1.capacity() % 4 == 0);
        for (int i = 0; i < d1.capacity(); i += 4) {
            double t0 = d1.get(i) - d2.get(i);
            double t1 = d1.get(i + 1) - d2.get(i + 1);
            double t2 = d1.get(i + 2) - d2.get(i + 2);
            double t3 = d1.get(i + 3) - d2.get(i + 3);
            totalCost += t0 * t0 + t1 * t1 + t2 * t2 + t3 * t3;
            if (totalCost > best) {
                break;
            }
        }
        return totalCost;
    }

    private int naiveNearestNeighbor(FloatBuffer vec, int laplacian, CvSURFPoint[] modelKeypoints, FloatBuffer[] modelDescriptors) {
        int neighbor = -1;
        double d, dist1 = 1e6, dist2 = 1e6;

        for (int i = 0; i < modelDescriptors.length; i++) {
            CvSURFPoint kp = modelKeypoints[i];
            FloatBuffer mvec = modelDescriptors[i];
            if (laplacian != kp.laplacian()) {
                continue;
            }
            d = compareSURFDescriptors(vec, mvec, dist2);
            if (d < dist1) {
                dist2 = dist1;
                dist1 = d;
                neighbor = i;
            } else if (d < dist2) {
                dist2 = d;
            }
        }
        if (dist1 < settings.distanceThreshold * dist2) {
            return neighbor;
        }
        return -1;
    }

    private void findPairs(CvSURFPoint[] objectKeypoints, FloatBuffer[] objectDescriptors, CvSURFPoint[] imageKeypoints, FloatBuffer[] imageDescriptors) {
        for (int i = 0; i < objectDescriptors.length; i++) {
            CvSURFPoint kp = objectKeypoints[i];
            FloatBuffer descriptor = objectDescriptors[i];
            int nearestNeighbor = naiveNearestNeighbor(descriptor, kp.laplacian(), imageKeypoints, imageDescriptors);
            if (nearestNeighbor >= 0) {
                ptpairs.add(i);
                ptpairs.add(nearestNeighbor);
            }
        }
    }

    private void flannFindPairs(FloatBuffer[] objectDescriptors, FloatBuffer[] imageDescriptors) {
        int length = objectDescriptors[0].capacity();

        if (imageMat.rows() < imageDescriptors.length) {
            imageMat.create(imageDescriptors.length, length, CV_32FC1);
        }
        int imageRows = imageMat.rows();
        imageMat.rows(imageDescriptors.length);

        // copy descriptors
        FloatBuffer objectBuf = objectMat.getFloatBuffer();
        for (int i = 0; i < objectDescriptors.length; i++) {
            objectBuf.put(objectDescriptors[i]);
        }

        FloatBuffer imageBuf = imageMat.getFloatBuffer();
        for (int i = 0; i < imageDescriptors.length; i++) {
            imageBuf.put(imageDescriptors[i]);
        }

        // find nearest neighbors using FLANN
        flannIndex.build(imageMat, indexParams, FLANN_DIST_L2);
        flannIndex.knnSearch(objectMat, indicesMat, distsMat, 2, searchParams);

        IntBuffer indicesBuf = indicesMat.getIntBuffer();
        FloatBuffer distsBuf = distsMat.getFloatBuffer();
        for (int i = 0; i < objectDescriptors.length; i++) {
            if (distsBuf.get(2 * i) < settings.distanceThreshold * distsBuf.get(2 * i + 1)) {
                ptpairs.add(i);
                ptpairs.add(indicesBuf.get(2 * i));
            }
        }
        imageMat.rows(imageRows);
    }

    /* a rough implementation for object location */
    private double[] locatePlanarObject(CvSURFPoint[] objectKeypoints, FloatBuffer[] objectDescriptors,
            CvSURFPoint[] imageKeypoints, FloatBuffer[] imageDescriptors, double[] srcCorners) {
        ptpairs.clear();
        if (settings.useFLANN) {
            flannFindPairs(objectDescriptors, imageDescriptors);
        } else {
            findPairs(objectKeypoints, objectDescriptors, imageKeypoints, imageDescriptors);
        }
        int n = ptpairs.size() / 2;
        logger.info(n + " matching pairs found");
        if (n < settings.matchesMin) {
            return null;
        }

        pt1.cols(n);
        pt2.cols(n);
        mask.cols(n);
        for (int i = 0; i < n; i++) {
            CvPoint2D32f p1 = objectKeypoints[ptpairs.get(2 * i)].pt();
            pt1.put(2 * i, p1.x());
            pt1.put(2 * i + 1, p1.y());
            CvPoint2D32f p2 = imageKeypoints[ptpairs.get(2 * i + 1)].pt();
            pt2.put(2 * i, p2.x());
            pt2.put(2 * i + 1, p2.y());
        }

        if (cvFindHomography(pt1, pt2, H, CV_RANSAC, settings.ransacReprojThreshold, mask) == 0) {
            return null;
        }
        if (cvCountNonZero(mask) < settings.matchesMin) {
            return null;
        }

        double[] h = H.get();
        double[] dstCorners = new double[srcCorners.length];
        for (int i = 0; i < srcCorners.length / 2; i++) {
            double x = srcCorners[2 * i], y = srcCorners[2 * i + 1];
            double Z = 1 / (h[6] * x + h[7] * y + h[8]);
            double X = (h[0] * x + h[1] * y + h[2]) * Z;
            double Y = (h[3] * x + h[4] * y + h[5]) * Z;
            dstCorners[2 * i] = X;
            dstCorners[2 * i + 1] = Y;
        }

        pt1.cols(objectDescriptors.length);
        pt2.cols(objectDescriptors.length);
        mask.cols(objectDescriptors.length);
        return dstCorners;
    }
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        jPanel1 = new javax.swing.JPanel();
        tabs = new javax.swing.JTabbedPane();
        jPanel2 = new javax.swing.JPanel();
        lvideo = new javax.swing.JButton();
        sproc = new javax.swing.JButton();
        vid1 = new javax.swing.JInternalFrame();
        vid2 = new javax.swing.JInternalFrame();
        lside = new javax.swing.JButton();
        jPanel3 = new javax.swing.JPanel();
        frontlabel = new javax.swing.JLabel();
        closeapp = new javax.swing.JButton();
        dest1 = new javax.swing.JTextField();
        busNo1 = new javax.swing.JTextField();
        jLabel2 = new javax.swing.JLabel();
        jLabel3 = new javax.swing.JLabel();
        rearlabel = new javax.swing.JLabel();
        busNo2 = new javax.swing.JTextField();
        jLabel1 = new javax.swing.JLabel();
        dest2 = new javax.swing.JTextField();
        jLabel4 = new javax.swing.JLabel();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);

        jPanel2.setCursor(new java.awt.Cursor(java.awt.Cursor.DEFAULT_CURSOR));

        lvideo.setText("Load Front Video");
        lvideo.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                lvideoActionPerformed(evt);
            }
        });

        sproc.setText("Start Processing");
        sproc.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                sprocActionPerformed(evt);
            }
        });

        vid1.setPreferredSize(new java.awt.Dimension(500, 500));
        vid1.setVisible(true);

        javax.swing.GroupLayout vid1Layout = new javax.swing.GroupLayout(vid1.getContentPane());
        vid1.getContentPane().setLayout(vid1Layout);
        vid1Layout.setHorizontalGroup(
            vid1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 484, Short.MAX_VALUE)
        );
        vid1Layout.setVerticalGroup(
            vid1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 470, Short.MAX_VALUE)
        );

        vid2.setPreferredSize(new java.awt.Dimension(500, 500));
        vid2.setVisible(true);

        javax.swing.GroupLayout vid2Layout = new javax.swing.GroupLayout(vid2.getContentPane());
        vid2.getContentPane().setLayout(vid2Layout);
        vid2Layout.setHorizontalGroup(
            vid2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 484, Short.MAX_VALUE)
        );
        vid2Layout.setVerticalGroup(
            vid2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 470, Short.MAX_VALUE)
        );

        lside.setText("Load SideView");
        lside.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                lsideActionPerformed(evt);
            }
        });

        javax.swing.GroupLayout jPanel2Layout = new javax.swing.GroupLayout(jPanel2);
        jPanel2.setLayout(jPanel2Layout);
        jPanel2Layout.setHorizontalGroup(
            jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel2Layout.createSequentialGroup()
                .addGap(76, 76, 76)
                .addComponent(vid1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(210, 210, 210)
                .addComponent(vid2, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addContainerGap(268, Short.MAX_VALUE))
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, jPanel2Layout.createSequentialGroup()
                .addGap(228, 228, 228)
                .addComponent(lvideo, javax.swing.GroupLayout.PREFERRED_SIZE, 175, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, 560, Short.MAX_VALUE)
                .addComponent(lside, javax.swing.GroupLayout.PREFERRED_SIZE, 170, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(421, 421, 421))
            .addGroup(jPanel2Layout.createSequentialGroup()
                .addGap(597, 597, 597)
                .addComponent(sproc, javax.swing.GroupLayout.PREFERRED_SIZE, 169, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addContainerGap(788, Short.MAX_VALUE))
        );
        jPanel2Layout.setVerticalGroup(
            jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel2Layout.createSequentialGroup()
                .addGap(23, 23, 23)
                .addGroup(jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(vid1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(vid2, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGap(18, 18, 18)
                .addGroup(jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(lvideo, javax.swing.GroupLayout.PREFERRED_SIZE, 35, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(lside, javax.swing.GroupLayout.PREFERRED_SIZE, 32, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGap(15, 15, 15)
                .addComponent(sproc, javax.swing.GroupLayout.PREFERRED_SIZE, 35, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(1116, 1116, 1116))
        );

        tabs.addTab("Detection", jPanel2);

        closeapp.setText("Close");
        closeapp.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                closeappActionPerformed(evt);
            }
        });

        dest1.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                dest1ActionPerformed(evt);
            }
        });

        jLabel2.setText("DESTINATION :");

        jLabel3.setText("BUS NUMBER :");

        jLabel1.setText("BUS NUMBER :");

        dest2.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                dest2ActionPerformed(evt);
            }
        });

        jLabel4.setText("DESTINATION :");

        javax.swing.GroupLayout jPanel3Layout = new javax.swing.GroupLayout(jPanel3);
        jPanel3.setLayout(jPanel3Layout);
        jPanel3Layout.setHorizontalGroup(
            jPanel3Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel3Layout.createSequentialGroup()
                .addGap(143, 143, 143)
                .addGroup(jPanel3Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                    .addComponent(jLabel3)
                    .addComponent(jLabel2))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                .addGroup(jPanel3Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(jPanel3Layout.createSequentialGroup()
                        .addComponent(busNo1, javax.swing.GroupLayout.PREFERRED_SIZE, 169, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, 445, Short.MAX_VALUE)
                        .addComponent(jLabel1))
                    .addGroup(jPanel3Layout.createSequentialGroup()
                        .addComponent(dest1, javax.swing.GroupLayout.PREFERRED_SIZE, 286, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, 323, Short.MAX_VALUE)
                        .addComponent(jLabel4)))
                .addGap(18, 18, 18)
                .addGroup(jPanel3Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(busNo2, javax.swing.GroupLayout.PREFERRED_SIZE, 169, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(dest2, javax.swing.GroupLayout.PREFERRED_SIZE, 286, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGap(338, 338, 338))
            .addGroup(jPanel3Layout.createSequentialGroup()
                .addGap(24, 24, 24)
                .addComponent(frontlabel, javax.swing.GroupLayout.PREFERRED_SIZE, 596, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(43, 43, 43)
                .addComponent(rearlabel, javax.swing.GroupLayout.PREFERRED_SIZE, 596, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addContainerGap(295, Short.MAX_VALUE))
            .addGroup(jPanel3Layout.createSequentialGroup()
                .addGap(591, 591, 591)
                .addComponent(closeapp, javax.swing.GroupLayout.PREFERRED_SIZE, 170, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addContainerGap(793, Short.MAX_VALUE))
        );
        jPanel3Layout.setVerticalGroup(
            jPanel3Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel3Layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(jPanel3Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(jPanel3Layout.createSequentialGroup()
                        .addComponent(frontlabel, javax.swing.GroupLayout.PREFERRED_SIZE, 404, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addGap(29, 29, 29)
                        .addGroup(jPanel3Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                            .addComponent(jLabel3)
                            .addComponent(busNo1, javax.swing.GroupLayout.PREFERRED_SIZE, 43, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addComponent(busNo2, javax.swing.GroupLayout.PREFERRED_SIZE, 43, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addComponent(jLabel1)))
                    .addComponent(rearlabel, javax.swing.GroupLayout.PREFERRED_SIZE, 404, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGap(19, 19, 19)
                .addGroup(jPanel3Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(dest1, javax.swing.GroupLayout.PREFERRED_SIZE, 43, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jLabel2)
                    .addComponent(dest2, javax.swing.GroupLayout.PREFERRED_SIZE, 43, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jLabel4))
                .addGap(43, 43, 43)
                .addComponent(closeapp, javax.swing.GroupLayout.PREFERRED_SIZE, 43, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addContainerGap(1107, Short.MAX_VALUE))
        );

        tabs.addTab("Matching", jPanel3);

        javax.swing.GroupLayout jPanel1Layout = new javax.swing.GroupLayout(jPanel1);
        jPanel1.setLayout(jPanel1Layout);
        jPanel1Layout.setHorizontalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(tabs)
        );
        jPanel1Layout.setVerticalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(tabs)
        );

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addComponent(jPanel1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(0, 0, Short.MAX_VALUE))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addComponent(jPanel1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(0, 0, Short.MAX_VALUE))
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void closeappActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_closeappActionPerformed
        // TODO add your handling code here:
        File f;

        while (cnt > 0) {
            f = new File(cnt + "capture.jpg");
            f.delete();
            cnt--;
        }
        while (crop > 0) {
            f = new File(crop + "detected.png");
            boolean s = f.delete();
            crop--;
        }
        cnt = 1;
        boolean b = true;
        while (b) {
            f = new File("C:\\Users\\Aparajita\\Documents\\NetBeansProjects\\VIAssist\\sideview\\" + cnt + "capture.jpg");
            b = f.isFile();
            if (b) {
                f.delete();
            }
            cnt++;
        }
        System.exit(0);
    }//GEN-LAST:event_closeappActionPerformed

    private void lvideoActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_lvideoActionPerformed
        // TODO add your handling code here:
        String vlcpath = "C:\\Program Files\\VideoLAN\\VLC";
        mediapath = "";

        File ourFile;

        ourFileSelector.setFileSelectionMode(JFileChooser.FILES_ONLY);
        ourFileSelector.showSaveDialog(null);
        ourFile = ourFileSelector.getSelectedFile();
        try {
            mediapath = ourFile.getAbsolutePath();
            new MediaPlayer1(vlcpath, mediapath).run();
        } catch (Exception e) {
        }


        //sproc.setEnabled(true);
    }//GEN-LAST:event_lvideoActionPerformed

    private void sprocActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_sprocActionPerformed
        // TODO add your handling code here:
        if (evt.getSource() == sproc) {

            // check if we are at the end of tabs //

            if (tabs.getSelectedIndex() == tabs.getTabCount() - 1) {
                tabs.setSelectedIndex(0);
            } else {
                tabs.setSelectedIndex(tabs.getSelectedIndex() + 1);
            }
        }

        sproc.setEnabled(false);
        //Detection
        FrameGrabber grabber = new OpenCVFrameGrabber(mediapath);
        if (grabber == null) {
            System.out.println("!!! Failed OpenCVFrameGrabber");
            return;
        }
        //CanvasFrame canvas = new CanvasFrame("video_demo");
        try {
            grabber.start();
            IplImage frame = null;

            int c = 1;
            cnt = 1;
            crop = 0;
            int frame_counter = 1;
            while (true) {
                frame = grabber.grab();
                if (frame == null) {
                    System.out.println("!!! Failed grab");
                    break;
                }
                if ((frame_counter % 5) == 0) {
                    //canvas.showImage(frame);
                    //canvas.setVisible(false);
                    cvSaveImage((c++) + "capture.jpg", frame);
                    String objectFilename = "229 cropped.jpg";
                    String sceneFilename = "C:\\Users\\Aparajita\\Documents\\NetBeansProjects\\VIAssist\\" + (cnt) + "capture.jpg";
                    IplImage object = cvLoadImage(objectFilename, CV_LOAD_IMAGE_GRAYSCALE);
                    IplImage image = cvLoadImage(sceneFilename, CV_LOAD_IMAGE_GRAYSCALE);
                    if (object == null || image == null) {
                        System.err.println("Can not load " + objectFilename + " and/or " + sceneFilename);
                        System.exit(-1);
                    }

                    IplImage objectColor = IplImage.create(object.width(), object.height(), 8, 3);
                    cvCvtColor(object, objectColor, CV_GRAY2BGR);

                    IplImage correspond = IplImage.create(image.width(), object.height() + image.height(), 8, 1);
                    cvSetImageROI(correspond, cvRect(0, 0, object.width(), object.height()));
                    cvCopy(object, correspond);
                    cvSetImageROI(correspond, cvRect(0, object.height(), correspond.width(), correspond.height()));
                    cvCopy(image, correspond);
                    cvResetImageROI(correspond);

                    ObjectFinder.Settings settings = new ObjectFinder.Settings();
                    settings.objectImage = object;
                    settings.useFLANN = true;
                    settings.ransacReprojThreshold = 5;
                    ObjectFinder finder = new ObjectFinder(settings);

                    long start = System.currentTimeMillis();
                    double[] dst_corners = finder.find(image);
                    System.out.println("Finding time = " + (System.currentTimeMillis() - start) + " ms");
                    int mpt;
                    mpt = ptpairs.size() / 2;
                    int arr[] = new int[10];
                    if (dst_corners != null) {
                        for (int i = 0; i < 4; i++) {
                            int j = (i + 1) % 4;
                            int x1 = (int) Math.round(dst_corners[2 * i]);
                            int y1 = (int) Math.round(dst_corners[2 * i + 1]);
                            int x2 = (int) Math.round(dst_corners[2 * j]);
                            int y2 = (int) Math.round(dst_corners[2 * j + 1]);
                            if (i == 0) {
                                arr[i] = x1;
                                arr[i + 1] = y1;
                                arr[i + 2] = x2;
                                arr[i + 3] = y2;
                            }
                            cvLine(correspond, cvPoint(x1, y1 + object.height()), cvPoint(x2, y2 + object.height()), CvScalar.RED, 8, 8, 0);
                        }
                    }
                    if (detect) {
                        if (mpt > 31) {
                            for (int i = 0; i < 5; i++) {
                                Thread.sleep(500);
                                java.awt.Toolkit.getDefaultToolkit().beep();
                            }
                            detect = false;
                        }
                    }
                    if (mpt > 31) {

                        IplImage src = cvLoadImage("C:\\Users\\Aparajita\\Documents\\NetBeansProjects\\VIAssist\\" + (cnt) + "capture.jpg", CV_LOAD_IMAGE_COLOR);
                        CvRect rect = new CvRect();
                        System.out.println(arr[0] + "" + arr[1]);
                        rect.x(arr[0] - 100);
                        rect.y(arr[1] - 100);
                        rect.width(400);
                        rect.height(600);
                        //cvSetImageROI(image, rect);
                        IplImage detected = cvCreateImage(cvGetSize(image), image.depth(), image.nChannels());
                        cvCopy(image, detected);
                        cvSaveImage((++crop) + "detected.png", detected);
                        // if(crop==5)
                        //break; 
                    }
                    cnt++;
                    for (int i = 0; i < finder.ptpairs.size(); i += 2) {
                        CvPoint2D32f pt1 = finder.objectKeypoints[finder.ptpairs.get(i)].pt();
                        CvPoint2D32f pt2 = finder.imageKeypoints[finder.ptpairs.get(i + 1)].pt();
                        cvLine(correspond, cvPointFrom32f(pt1), cvPoint(Math.round(pt2.x()), Math.round(pt2.y() + object.height())), CvScalar.WHITE, 1, 8, 0);
                    }

                    //CanvasFrame objectFrame = new CanvasFrame("Object");
                    //CanvasFrame correspondFrame = new CanvasFrame("Object Correspond");

                    //correspondFrame.showImage(correspond);
                    //correspondFrame.setVisible(false);
                    for (int i = 0; i < finder.objectKeypoints.length; i++) {
                        CvSURFPoint r = finder.objectKeypoints[i];
                        CvPoint center = cvPointFrom32f(r.pt());
                        int radius = Math.round(r.size() * 1.2f / 9 * 2);
                        cvCircle(objectColor, center, radius, CvScalar.RED, 1, 8, 0);
                    }
                    // objectFrame.showImage(objectColor);

                    //objectFrame.waitKey(1);
                    //objectFrame.setVisible(false);

                    //objectFrame.dispose();
                    //correspondFrame.dispose();
                }
                frame_counter++;
            }
        } catch (Exception e) {
        }

        //Template Matching
        double coef = 0;
        int tempcpy = 0;
        if (crop == 0) {

            JOptionPane.showMessageDialog(null, "Bus Not Detected", "Dialog", JOptionPane.ERROR_MESSAGE);
            //System.exit(0);
        }
        IplImage result;
        //System.out.println(crop);
        boolean chk = false;
        int t = 1;
        int temp = 1;
        IplImage src = cvLoadImage((t) + "detected.png", 0);
        IplImage tmp = cvLoadImage((temp) + ".jpg", 0);
        result = cvCreateImage(cvSize(src.width() - tmp.width() + 1, src.height() - tmp.height() + 1), IPL_DEPTH_32F, 1);
        while (t <= crop) {
            src = cvLoadImage((t) + "detected.png", 0);
            temp = 1;
            while (temp < 5) {
                tmp = cvLoadImage((temp) + ".jpg", 0);
                result = cvCreateImage(cvSize(src.width() - tmp.width() + 1, src.height() - tmp.height() + 1), IPL_DEPTH_32F, 1);

                cvZero(result);

                //Match Template Function from OpenCV
                //cvMatchTemplate(src, tmp, result, CV_TM_CCOEFF_NORMED);
                cvMatchTemplate(src, tmp, result, CV_TM_CCORR_NORMED);
                //cvMatchTemplate(src, tmp, result, CV_TM_SQDIFF_NORMED);
                double[] min_val = new double[2];
                double[] max_val = new double[2];
                int minLoc[] = new int[2];
                int maxLoc[] = new int[2];

                //Get the Max or Min Correlation Value		
                cvMinMaxLoc(result, min_val, max_val, minLoc, maxLoc, null);

                System.out.println(Arrays.toString(min_val));
                System.out.println(Arrays.toString(max_val));

                if (max_val[0] > 0.91) {
                    if (max_val[0] > coef) {
                        coef = max_val[0];
                        tempcpy = temp;
                    }
                    System.out.println("Match successful");
                    System.out.println(t);
                    System.out.println(max_val[0]);
                    //CvPoint point = new CvPoint();
                    int point[] = new int[2];
                    point[0] = maxLoc[0] + tmp.width();
                    point[1] = maxLoc[1] + tmp.height();

                    //cvRectangle(src, maxLoc, point, CvScalar.WHITE, 2, 8, 0);
                    //Draw a Rectangle for Matched Region
                    //cvShowImage("Matched Template", src);
                    cvWaitKey(2000);
                    chk = true;
                    BufferedImage img = null;
                    try {
                        img = src.getBufferedImage();
                        Image dimg = img.getScaledInstance(720, 404, Image.SCALE_SMOOTH);
                        ImageIcon imageIcon = new ImageIcon(dimg);
                        frontlabel.setIcon(imageIcon);
                    } catch (Exception e) {
                    }
                    //break;
                } else {
                    System.out.println("No correct match");

                }
                //cvReleaseImage(src);
                //cvReleaseImage(tmp);
                //cvReleaseImage(result);
                System.out.println(temp);
                temp++;
            }
            /*if (chk) {
            break;
            }*/
            t++;
        }
        if (chk) {

            try {
                System.out.println("tempcpy="+tempcpy);
                Class.forName("org.sqlite.JDBC");
                c = DriverManager.getConnection("jdbc:sqlite:test.db");
                System.out.println("Opened database successfully");
                c.setAutoCommit(false);
                System.out.println(temp);
                stmt = c.createStatement();
                stmt = c.createStatement();
                rs = stmt.executeQuery("SELECT B FROM VIAssist WHERE A='" + tempcpy + "';");
                String B = rs.getString("B");
                System.out.println(B);
                busNo1.setText(B);

                rs = stmt.executeQuery("SELECT C FROM VIAssist WHERE A='" + tempcpy + "';");
                String C = rs.getString("C");
                System.out.println(C);
                dest1.setText(C);
                rs.close();
                stmt.close();
                c.close();
            } catch (Exception e) {
            }


            //Feedback 
            Voice voice;
            VoiceManager vm = VoiceManager.getInstance();
            voice = vm.getVoice(VOICENAME);
            voice.allocate();
            String B = busNo1.getText();
            voice.speak(B);
            String C = dest1.getText();
            voice.speak("Destination is" + C);
            if (chk) {
                System.out.println(tempcpy);
                boolean b = sideview.rear(tempcpy, mediapath1);
                if(b)
                    JOptionPane.showMessageDialog(null, "Same Bus!", "MESSAGE", JOptionPane.ERROR_MESSAGE);
                System.out.println(b);
                if (!b) {

                    tempcpy = sideview.rear();
                    System.out.println("matched template"+tempcpy);
                    if (tempcpy != 0) {
                        try {
                            Class.forName("org.sqlite.JDBC");
                            c = DriverManager.getConnection("jdbc:sqlite:test.db");
                            System.out.println("Opened database successfully");
                            c.setAutoCommit(false);
                            System.out.println(temp);
                            stmt = c.createStatement();
                            stmt = c.createStatement();
                            rs = stmt.executeQuery("SELECT B FROM Rear WHERE A='" + tempcpy + "';");
                            B = rs.getString("B");
                            System.out.println(B);
                            busNo2.setText(B);

                            rs = stmt.executeQuery("SELECT C FROM Rear WHERE A='" + tempcpy + "';");
                            C = rs.getString("C");
                            System.out.println(C);
                            dest2.setText(C);
                            rs.close();
                            stmt.close();
                            c.close();
                        } catch (Exception e) {
                        }
                        //Feedback 
                        voice = vm.getVoice(VOICENAME);
                        voice.allocate();

                        B = busNo2.getText();
                        voice.speak(B);
                        C = dest2.getText();
                        voice.speak("Destination is" + C);
                    } else {
                        JOptionPane.showMessageDialog(null, "Rear Number not detected", "Error", JOptionPane.ERROR_MESSAGE);
                    }
                }
            } else {
                JOptionPane.showMessageDialog(null, "Bus not detected", "Error", JOptionPane.ERROR_MESSAGE);
            }
        } else {
            JOptionPane.showMessageDialog(null, "Bus Number Not Recognized", "Error", JOptionPane.ERROR_MESSAGE);
        }
    }//GEN-LAST:event_sprocActionPerformed

    private void lsideActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_lsideActionPerformed
        String vlcpath = "C:\\Program Files\\VideoLAN\\VLC";
        mediapath1 = "";

        File ourFile;

        ourFileSelector.setFileSelectionMode(JFileChooser.FILES_ONLY);
        ourFileSelector.showSaveDialog(null);
        ourFile = ourFileSelector.getSelectedFile();
        try {
            mediapath1 = ourFile.getAbsolutePath();
            new MediaPlayer1(vlcpath, mediapath1, true).run();
        } catch (Exception e) {
        }


        sproc.setEnabled(true);
    }//GEN-LAST:event_lsideActionPerformed

    private void dest1ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_dest1ActionPerformed
        // TODO add your handling code here:
    }//GEN-LAST:event_dest1ActionPerformed

    private void dest2ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_dest2ActionPerformed
        // TODO add your handling code here:
    }//GEN-LAST:event_dest2ActionPerformed

    public static void main(String args[]) {

        java.awt.EventQueue.invokeLater(new Runnable() {

            public void run() {
                new ObjectFinder().setVisible(true);
            }
        });
    }
    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JTextField busNo1;
    private javax.swing.JTextField busNo2;
    private javax.swing.JButton closeapp;
    private javax.swing.JTextField dest1;
    private javax.swing.JTextField dest2;
    public static javax.swing.JLabel frontlabel;
    private javax.swing.JLabel jLabel1;
    private javax.swing.JLabel jLabel2;
    private javax.swing.JLabel jLabel3;
    private javax.swing.JLabel jLabel4;
    private javax.swing.JPanel jPanel1;
    private javax.swing.JPanel jPanel2;
    private javax.swing.JPanel jPanel3;
    private javax.swing.JButton lside;
    private javax.swing.JButton lvideo;
    public static javax.swing.JLabel rearlabel;
    private javax.swing.JButton sproc;
    private javax.swing.JTabbedPane tabs;
    public static javax.swing.JInternalFrame vid1;
    public static javax.swing.JInternalFrame vid2;
    // End of variables declaration//GEN-END:variables
}
