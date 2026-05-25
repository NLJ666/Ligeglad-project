import javafx.application.Application;
import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;
import java.util.Random;

public class SimpleRandomGenerator extends Application {

    private Label numberLabel = new Label("0");
    private Label historyLabel = new Label("历史记录: 无");
    private StringBuilder history = new StringBuilder();
    private Random random = new Random();
    private int count = 0;

    @Override
    public void start(Stage stage) {
        numberLabel.setStyle("-fx-font-size: 48px; -fx-text-fill: red;");

        Button genBtn = new Button("生成随机数");
        genBtn.setOnAction(e -> {
            int num = random.nextInt(59) + 1;
            numberLabel.setText(String.valueOf(num));
            count++;
            history.append("第").append(count).append("次: ").append(num).append("\n");
            historyLabel.setText("历史记录:\n" + history.toString());
        });

        Button clearBtn = new Button("清空历史");
        clearBtn.setOnAction(e -> {
            history.setLength(0);
            count = 0;
            numberLabel.setText("0");
            historyLabel.setText("历史记录: 无");
        });

        VBox root = new VBox(10,
                new Label("1-59 随机数生成器"),
                numberLabel,
                genBtn,
                clearBtn,
                historyLabel
        );
        root.setPadding(new Insets(20));

        stage.setScene(new Scene(root, 300, 300));
        stage.setTitle("随机数生成器");
        stage.show();
    }

    public static void main(String[] args) {
        launch(args);
    }
}