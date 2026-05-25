import javafx.application.Application;
import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.stage.Stage;

public class HBoxButtonExample extends Application {

    @Override
    public void start(Stage primaryStage) {
        // 1. 创建两个按钮
        Button btnConfirm = new Button("确定");
        Button btnCancel = new Button("取消");

        // 2. 设置按钮最小尺寸
        btnConfirm.setMinWidth(100);
        btnCancel.setMinWidth(100);

        // 3. 设置按钮高度
        btnConfirm.setMinHeight(40);
        btnCancel.setMinHeight(40);

        // 4. 设置按钮字体
        btnConfirm.setStyle("-fx-font-size: 16px;");
        btnCancel.setStyle("-fx-font-size: 16px;");

        // 5. 创建 HBox 布局
        HBox hbox = new HBox(10); // 10 是按钮之间的间距
        hbox.getChildren().addAll(btnConfirm, btnCancel);

        // 6. 设置按钮占据整个 HBox 宽度
        HBox.setHgrow(btnConfirm, Priority.ALWAYS);
        HBox.setHgrow(btnCancel, Priority.ALWAYS);

        // 7. 设置按钮的填充宽度
        btnConfirm.setMaxWidth(Double.MAX_VALUE);
        btnCancel.setMaxWidth(Double.MAX_VALUE);

        // 8. 设置 HBox 的内边距
        hbox.setPadding(new Insets(20));

        // 9. 创建场景
        Scene scene = new Scene(hbox, 400, 100);

        // 10. 设置舞台
        primaryStage.setTitle("HBox示例                                                                       xx");
        primaryStage.setScene(scene);

        // 11. 显示舞台
        primaryStage.show();

        primaryStage.setMinWidth(300);
        primaryStage.setMinHeight(150);
    }

    public static void main(String[] args) {
        launch(args);
    }
}