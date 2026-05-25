import javafx.application.Application;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.VBox;
import javafx.scene.media.Media;
import javafx.scene.media.MediaPlayer;
import javafx.scene.media.MediaView;
import javafx.stage.Stage;
import javafx.scene.control.Label;
import javafx.scene.control.Slider;
import java.io.File;
import java.net.URI;

public class VideoPlayerApp extends Application {

    private MediaPlayer mediaPlayer;
    private Button playButton;
    private Label statusLabel;
    private Slider volumeSlider;

    @Override
    public void start(Stage primaryStage) {
        try {
            // 创建状态标签
            statusLabel = new Label("准备播放视频");
            statusLabel.setStyle("-fx-font-size: 14px; -fx-text-fill: #666;");

            // 视频文件路径 - 修改为你实际的视频文件路径
            String videoPath = "videos/video.mp4"; // 视频放在项目根目录

            // 检查视频文件是否存在
            File videoFile = new File(videoPath);
            if (!videoFile.exists()) {
                // 尝试其他路径
                videoPath = "src/video.mp4";
                videoFile = new File(videoPath);

                if (!videoFile.exists()) {
                    statusLabel.setText("错误: 视频文件未找到！请将视频文件放在项目根目录");
                    showErrorUI(primaryStage);
                    return;
                }
            }

            // 创建媒体对象
            URI uri = videoFile.toURI();
            System.out.println("视频文件URI: " + uri.toString());

            Media media = new Media(uri.toString());

            // 创建媒体播放器
            mediaPlayer = new MediaPlayer(media);

            // 创建媒体视图
            MediaView mediaView = new MediaView(mediaPlayer);
            mediaView.setFitWidth(800);
            mediaView.setFitHeight(450);

            // 音量控制滑块
            volumeSlider = new Slider(0, 100, 50);
            volumeSlider.setPrefWidth(150);
            volumeSlider.valueProperty().addListener((obs, oldVal, newVal) -> {
                if (mediaPlayer != null) {
                    mediaPlayer.setVolume(newVal.doubleValue() / 100.0);
                }
            });

            // 创建播放/暂停按钮
            playButton = new Button("▶ 播放");
            playButton.setStyle("-fx-font-size: 16px; -fx-padding: 10px 30px; -fx-background-color: #4CAF50; -fx-text-fill: white;");
            playButton.setOnAction(event -> togglePlayPause());

            // 创建按钮和音量控制的水平容器
            HBox controlsBox = new HBox(20, playButton, new Label("音量:"), volumeSlider);
            controlsBox.setAlignment(Pos.CENTER);
            controlsBox.setStyle("-fx-padding: 20px;");

            // 创建控制面板（垂直排列控制框和状态标签）
            VBox controlPanel = new VBox(10, controlsBox, statusLabel);
            controlPanel.setAlignment(Pos.CENTER);
            controlPanel.setStyle("-fx-padding: 20px; -fx-background-color: #f8f8f8;");

            // 将MediaView放入容器中
            StackPane videoContainer = new StackPane(mediaView);
            videoContainer.setStyle("-fx-background-color: black; -fx-padding: 10px;");
            videoContainer.setAlignment(Pos.CENTER);

            // 使用BorderPane布局，将视频放在中间，控制面板放在底部
            BorderPane root = new BorderPane();
            root.setCenter(videoContainer);
            root.setBottom(controlPanel);

            // 设置事件监听器
            mediaPlayer.setOnReady(() -> {
                statusLabel.setText("视频已加载，准备播放");
                playButton.setDisable(false);
            });

            mediaPlayer.setOnPlaying(() -> {
                statusLabel.setText("正在播放...");
                playButton.setText("⏸ 暂停");
            });

            mediaPlayer.setOnPaused(() -> {
                statusLabel.setText("已暂停");
                playButton.setText("▶ 播放");
            });

            mediaPlayer.setOnEndOfMedia(() -> {
                statusLabel.setText("播放完成");
                playButton.setText("▶ 播放");
            });

            mediaPlayer.setOnError(() -> {
                statusLabel.setText("播放错误: " + mediaPlayer.getError().getMessage());
                playButton.setDisable(true);
            });

            // 创建场景
            Scene scene = new Scene(root, 850, 600);

            // 设置窗口属性
            primaryStage.setTitle("JavaFX 视频播放器 - " + videoFile.getName());
            primaryStage.setScene(scene);
            primaryStage.show();

            // 窗口关闭时释放资源
            primaryStage.setOnCloseRequest(event -> {
                if (mediaPlayer != null) {
                    mediaPlayer.stop();
                    mediaPlayer.dispose();
                }
            });

        } catch (Exception e) {
            System.err.println("程序启动错误: " + e.getMessage());
            e.printStackTrace();
            statusLabel.setText("程序启动错误: " + e.getMessage());
            showErrorUI(primaryStage);
        }
    }

    // 切换播放/暂停
    private void togglePlayPause() {
        if (mediaPlayer == null) return;

        MediaPlayer.Status status = mediaPlayer.getStatus();

        switch (status) {
            case READY:
            case PAUSED:
            case STOPPED:
                mediaPlayer.play();
                break;
            case PLAYING:
                mediaPlayer.pause();
                break;
        }
    }

    // 显示错误界面
    private void showErrorUI(Stage primaryStage) {
        Label errorLabel = new Label("无法加载视频播放器");
        errorLabel.setStyle("-fx-font-size: 18px; -fx-text-fill: red;");

        Button closeButton = new Button("关闭");
        closeButton.setOnAction(e -> primaryStage.close());

        VBox errorBox = new VBox(20, errorLabel, closeButton);
        errorBox.setAlignment(Pos.CENTER);
        errorBox.setStyle("-fx-padding: 50px;");

        Scene errorScene = new Scene(errorBox, 400, 300);
        primaryStage.setTitle("错误");
        primaryStage.setScene(errorScene);
        primaryStage.show();
    }

    public static void main(String[] args) {
        // 处理JavaFX的警告
        System.setProperty("prism.order", "sw");
        launch(args);
    }
}