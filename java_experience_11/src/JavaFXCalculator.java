import javafx.application.Application;
import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.TextField;
import javafx.scene.layout.*;
import javafx.stage.Stage;

public class JavaFXCalculator extends Application {

    @Override
    public void start(Stage primaryStage) {
        primaryStage.setTitle("JavaFX 计算器");

        // 主布局：垂直盒子
        VBox root = new VBox(10);
        root.setPadding(new Insets(15));
        root.setStyle("-fx-background-color: #f5f5f5;");

        // 1. 显示区域（TextField）
        TextField display = new TextField("0");
        display.setEditable(false);
        display.setPrefHeight(50);
        display.setStyle("-fx-font-size: 24px; -fx-alignment: center-right; -fx-background-color: #ffffff; -fx-border-color: #cccccc; -fx-border-radius: 5;");
        display.setFocusTraversable(false);

        // 2. 功能按钮行（科学、MC、MR、M+、M-、C、AC）
        HBox funcRow = new HBox(5);
        funcRow.setPadding(new Insets(5, 0, 5, 0));
        funcRow.setStyle("-fx-alignment: center;"); // 可选：让按钮水平居中

// 为每个功能按钮设置最小宽度，并让它们拉伸
        Button btnScience = createButton("科学", "#666666", "#ffffff", 14);
        Button btnMC = createButton("MC", "#e0e0e0", "#333333", 14);
        Button btnMR = createButton("MR", "#e0e0e0", "#333333", 14);
        Button btnMPlus = createButton("M+", "#e0e0e0", "#333333", 14);
        Button btnMMinus = createButton("M-", "#e0e0e0", "#333333", 14);
        Button btnC = createButton("C", "#ff4444", "#ffffff", 14);
        Button btnAC = createButton("AC", "#ff4444", "#ffffff", 14);

// 为每个按钮设置最小宽度（根据文字长度调整，比如“科学”需要更宽）
        btnScience.setMinWidth(60);
        btnMC.setMinWidth(40);
        btnMR.setMinWidth(40);
        btnMPlus.setMinWidth(40);
        btnMMinus.setMinWidth(40);
        btnC.setMinWidth(40);
        btnAC.setMinWidth(40);

        // 将按钮添加到 HBox，并设置 HBox 的拉伸策略（让按钮均匀分布并拉伸）
        funcRow.getChildren().addAll(btnScience, btnMC, btnMR, btnMPlus, btnMMinus, btnC, btnAC);
        HBox.setHgrow(btnScience, Priority.ALWAYS);
        HBox.setHgrow(btnMC, Priority.ALWAYS);
        HBox.setHgrow(btnMR, Priority.ALWAYS);
        HBox.setHgrow(btnMPlus, Priority.ALWAYS);
        HBox.setHgrow(btnMMinus, Priority.ALWAYS);
        HBox.setHgrow(btnC, Priority.ALWAYS);
        HBox.setHgrow(btnAC, Priority.ALWAYS);

        // 3. 数字与操作符网格（4行5列）
        GridPane grid = new GridPane();
        grid.setHgap(5); // 水平间距
        grid.setVgap(5); // 垂直间距
        grid.setPadding(new Insets(10, 0, 10, 0));
        grid.setStyle("-fx-alignment: center;");

        // 定义按钮文字（按行排列）
        String[][] buttonTexts = {
                {"7", "8", "9", "/", "sqrt"},
                {"4", "5", "6", "*", "x²"},
                {"1", "2", "3", "-", "1/x"},
                {"0", ".", "=", "+", "%"}
        };

        // 定义按钮颜色（背景色、文字色）
        String[][] buttonStyles = {
                {"#f0f0f0", "#000000"}, {"#f0f0f0", "#000000"}, {"#f0f0f0", "#000000"}, {"#f0f0f0", "#000000"}, {"#f0f0f0", "#000000"},
                {"#f0f0f0", "#000000"}, {"#f0f0f0", "#000000"}, {"#f0f0f0", "#000000"}, {"#f0f0f0", "#000000"}, {"#f0f0f0", "#000000"},
                {"#f0f0f0", "#000000"}, {"#f0f0f0", "#000000"}, {"#f0f0f0", "#000000"}, {"#f0f0f0", "#000000"}, {"#f0f0f0", "#000000"},
                {"#f0f0f0", "#000000"}, {"#f0f0f0", "#000000"}, {"#007bff", "#ffffff"}, {"#f0f0f0", "#000000"}, {"#f0f0f0", "#000000"}
        };

        // 遍历行和列，创建按钮并添加到 GridPane
        for (int row = 0; row < buttonTexts.length; row++) {
            for (int col = 0; col < buttonTexts[row].length; col++) {
                String text = buttonTexts[row][col];
                String bgColor = buttonStyles[row][0];
                String textColor = buttonStyles[row][1];
                int fontSize = 14;

                Button btn = createButton(text, bgColor, textColor, fontSize);
                btn.setMinWidth(60); // 设置最小宽度，确保文字显示

                // 将按钮添加到 GridPane
                grid.add(btn, col, row);

                // 让 GridPane 的列拉伸（使列宽均匀分布并随窗口拉伸）
                GridPane.setHgrow(btn, Priority.ALWAYS);
            }
        }

        // 额外设置：让 GridPane 的列宽度均匀分布（可选，增强拉伸效果）
        grid.getColumnConstraints().clear();
        for (int i = 0; i < 5; i++) { // 5列
            ColumnConstraints colConst = new ColumnConstraints();
            colConst.setHgrow(Priority.ALWAYS); // 列随窗口拉伸
            colConst.setPercentWidth(20); // 每列占20%宽度（5列×20%=100%）
            grid.getColumnConstraints().add(colConst);
        }

        // 4. 组装主布局
        root.getChildren().addAll(display, funcRow, grid);

        // 5. 设置场景和显示
        Scene scene = new Scene(root, 320, 480);
        primaryStage.setScene(scene);
        primaryStage.show();
    }

    private Button createButton(String text, String bgColor, String textColor, int fontSize) {
        Button btn = new Button(text);
        btn.setPrefSize(50, 50);
        btn.setStyle(
                "-fx-background-color: " + bgColor + "; " +
                        "-fx-text-fill: " + textColor + "; " +
                        "-fx-font-size: " + fontSize + "px; " +
                        "-fx-min-width: 60px; " + // 显式设置最小宽度
                        "-fx-min-height: 40px; " +
                        "-fx-alignment: center;" +
                        "-fx-font-weight: bold; " +
                        "-fx-background-radius: 5; " +
                        "-fx-border-radius: 5; " +
                        "-fx-border-color: #cccccc;"
        );
        return btn;
    }
    public static void main(String[] args) {
        launch(args);
    }
}