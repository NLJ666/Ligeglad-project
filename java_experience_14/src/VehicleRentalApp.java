import javafx.application.Application;
import javafx.beans.property.SimpleDoubleProperty;
import javafx.beans.property.SimpleStringProperty;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import javafx.stage.Stage;

// ================= 租赁接口 =================
interface Rentable {
    void rent(String customerName);
    void returnVehicle();
    double calculateRent(int days);
    boolean isAvailable();
    String getCustomerName();
}

// ================= 车辆父类 =================
abstract class Vehicle {
    private String licensePlate;
    private String brand;
    private int productionYear;
    private double dailyRent;
    protected boolean isRented;

    public Vehicle(String licensePlate, String brand, int productionYear, double dailyRent) {
        this.licensePlate = licensePlate;
        this.brand = brand;
        this.productionYear = productionYear;
        this.dailyRent = dailyRent;
        this.isRented = false;
    }

    public abstract double calculateMaintenanceCost();

    public String getLicensePlate() { return licensePlate; }
    public String getBrand() { return brand; }
    public int getProductionYear() { return productionYear; }
    public double getDailyRent() { return dailyRent; }

    public boolean isRented() { return isRented; }
    public void setRented(boolean rented) { isRented = rented; }
}

// ================= 轿车 =================
class Car extends Vehicle implements Rentable {
    private int seats;
    private String fuelType;
    private String customerName;

    public Car(String licensePlate, String brand, int productionYear, double dailyRent, int seats, String fuelType) {
        super(licensePlate, brand, productionYear, dailyRent);
        this.seats = seats;
        this.fuelType = fuelType;
    }

    @Override
    public double calculateMaintenanceCost() {
        return getDailyRent() * 0.1;
    }

    @Override
    public void rent(String customerName) {
        if (!isRented()) {
            setRented(true);
            this.customerName = customerName;
        }
    }

    @Override
    public void returnVehicle() {
        setRented(false);
        this.customerName = null;
    }

    @Override
    public double calculateRent(int days) {
        return getDailyRent() * days;
    }

    @Override
    public boolean isAvailable() {
        return !isRented();
    }

    public int getSeats() { return seats; }
    public String getFuelType() { return fuelType; }
    @Override public String getCustomerName() { return customerName; }
}

// ================= 货车 =================
class Truck extends Vehicle implements Rentable {
    private double loadCapacity;
    private boolean hasInsurance;
    private String customerName;

    public Truck(String licensePlate, String brand, int productionYear, double dailyRent, double loadCapacity, boolean hasInsurance) {
        super(licensePlate, brand, productionYear, dailyRent);
        this.loadCapacity = loadCapacity;
        this.hasInsurance = hasInsurance;
    }

    @Override
    public double calculateMaintenanceCost() {
        return getDailyRent() * 0.2;
    }

    @Override
    public void rent(String customerName) {
        if (!isRented()) {
            setRented(true);
            this.customerName = customerName;
        }
    }

    @Override
    public void returnVehicle() {
        setRented(false);
        this.customerName = null;
    }

    @Override
    public double calculateRent(int days) {
        return getDailyRent() * days;
    }

    @Override
    public boolean isAvailable() {
        return !isRented();
    }

    public double getLoadCapacity() { return loadCapacity; }
    public boolean isHasInsurance() { return hasInsurance; }
    @Override public String getCustomerName() { return customerName; }
}

// ================= 客车 =================
class Bus extends Vehicle implements Rentable {
    private int passengerCapacity;
    private boolean hasWifi;
    private String customerName;

    public Bus(String licensePlate, String brand, int productionYear, double dailyRent, int passengerCapacity, boolean hasWifi) {
        super(licensePlate, brand, productionYear, dailyRent);
        this.passengerCapacity = passengerCapacity;
        this.hasWifi = hasWifi;
    }

    @Override
    public double calculateMaintenanceCost() {
        return getDailyRent() * 0.15;
    }

    @Override
    public void rent(String customerName) {
        if (!isRented()) {
            setRented(true);
            this.customerName = customerName;
        }
    }

    @Override
    public void returnVehicle() {
        setRented(false);
        this.customerName = null;
    }

    @Override
    public double calculateRent(int days) {
        return getDailyRent() * days;
    }

    @Override
    public boolean isAvailable() {
        return !isRented();
    }

    public int getPassengerCapacity() { return passengerCapacity; }
    public boolean isHasWifi() { return hasWifi; }
    @Override public String getCustomerName() { return customerName; }
}

// ================= JavaFX 主程序 =================
public class VehicleRentalApp extends Application {
    private ObservableList<Vehicle> vehicleList = FXCollections.observableArrayList();
    private TableView<Vehicle> vehicleTable;
    private Label statsLabel;
    private int rentDays = 5;

    public static void main(String[] args) {
        launch(args);
    }

    @Override
    public void start(Stage primaryStage) {
        primaryStage.setTitle("汽车租赁管理系统");
        initVehicles();
        initUI(primaryStage);
    }

    private void initVehicles() {
        vehicleList.add(new Car("京A12345", "Toyota", 2022, 300.0, 5, "汽油"));
        vehicleList.add(new Car("京B67890", "Tesla", 2023, 400.0, 5, "电动"));
        vehicleList.add(new Truck("沪C54321", "解放", 2020, 800.0, 3.5, true));
        vehicleList.add(new Bus("粤D98765", "宇通", 2021, 1200.0, 45, true));
    }

    private void initUI(Stage stage) {
        createVehicleTable();

        HBox buttonBox = new HBox(15);
        buttonBox.setPadding(new Insets(10));

        Button rentBtn = new Button("租赁车辆");
        Button returnBtn = new Button("归还车辆");
        Button calcBtn = new Button("计算租金");
        Button statsBtn = new Button("统计信息");

        buttonBox.getChildren().addAll(rentBtn, returnBtn, calcBtn, statsBtn);

        statsLabel = new Label("统计信息将显示在这里");
        statsLabel.setStyle("-fx-font-size: 14px; -fx-padding: 10px");

        VBox root = new VBox(10);
        root.setPadding(new Insets(15));
        root.getChildren().addAll(vehicleTable, buttonBox, statsLabel);

        rentBtn.setOnAction(e -> rentVehicle());
        returnBtn.setOnAction(e -> returnVehicle());
        calcBtn.setOnAction(e -> calculateAllRent());
        statsBtn.setOnAction(e -> showStatistics());

        Scene scene = new Scene(root, 900, 550);
        stage.setScene(scene);
        stage.show();
    }

    private void createVehicleTable() {
        vehicleTable = new TableView<>();
        vehicleTable.setItems(vehicleList);

        TableColumn<Vehicle, String> typeCol = new TableColumn<>("车辆类型");
        typeCol.setCellValueFactory(data -> {
            Vehicle v = data.getValue();
            if (v instanceof Car) return new SimpleStringProperty("轿车");
            if (v instanceof Truck) return new SimpleStringProperty("货车");
            if (v instanceof Bus) return new SimpleStringProperty("客车");
            return new SimpleStringProperty("未知");
        });

        TableColumn<Vehicle, String> plateCol = new TableColumn<>("车牌号");
        plateCol.setCellValueFactory(data -> new SimpleStringProperty(data.getValue().getLicensePlate()));

        TableColumn<Vehicle, String> brandCol = new TableColumn<>("品牌");
        brandCol.setCellValueFactory(data -> new SimpleStringProperty(data.getValue().getBrand()));

        TableColumn<Vehicle, String> statusCol = new TableColumn<>("状态");
        statusCol.setCellValueFactory(data -> {
            Rentable r = (Rentable) data.getValue();
            return new SimpleStringProperty(r.isAvailable() ? "可租赁" : "已出租");
        });

        TableColumn<Vehicle, String> customerCol = new TableColumn<>("租客");
        customerCol.setCellValueFactory(data -> {
            Rentable r = (Rentable) data.getValue();
            String name = r.getCustomerName();
            return new SimpleStringProperty(name == null ? "-" : name);
        });

        TableColumn<Vehicle, Number> rentCol = new TableColumn<>("日租金(元)");
        rentCol.setCellValueFactory(data -> new SimpleDoubleProperty(data.getValue().getDailyRent()));

        vehicleTable.getColumns().addAll(typeCol, plateCol, brandCol, rentCol, statusCol, customerCol);
    }

    private void rentVehicle() {
        Vehicle selected = vehicleTable.getSelectionModel().getSelectedItem();
        if (selected == null) {
            showAlert("请先选择一辆车辆！");
            return;
        }

        Rentable rentable = (Rentable) selected;
        if (!rentable.isAvailable()) {
            showAlert("该车辆已出租！");
            return;
        }

        TextInputDialog dialog = new TextInputDialog();
        dialog.setTitle("租赁车辆");
        dialog.setHeaderText("请输入租客姓名");
        dialog.showAndWait().ifPresent(name -> {
            rentable.rent(name);
            vehicleTable.refresh();
            showAlert("租赁成功！租客：" + name);
        });
    }

    private void returnVehicle() {
        Vehicle selected = vehicleTable.getSelectionModel().getSelectedItem();
        if (selected == null) {
            showAlert("请先选择一辆车辆！");
            return;
        }

        Rentable rentable = (Rentable) selected;
        if (rentable.isAvailable()) {
            showAlert("该车辆未出租！");
            return;
        }

        rentable.returnVehicle();
        vehicleTable.refresh();
        showAlert("车辆已归还！");
    }

    private void calculateAllRent() {
        StringBuilder sb = new StringBuilder("租金计算（" + rentDays + "天）：\n");
        double total = 0;

        for (Vehicle v : vehicleList) {
            Rentable r = (Rentable) v;
            // 只计算 已租出 的车辆
            if (!r.isAvailable()) {
                double rent = r.calculateRent(rentDays);
                double maintain = v.calculateMaintenanceCost();
                total += rent;

                sb.append(String.format("%s %s - 租金: %.1f元, 维护费: %.1f元%n",
                        v.getLicensePlate(), v.getBrand(), rent, maintain));
            }
        }

        sb.append(String.format("\n✅ 当前已租车辆总租金收入：%.1f元", total));
        showInfo("租金计算", sb.toString());
    }

    private void showStatistics() {
        int available = 0, rented = 0;
        double totalRevenue = 0;

        for (Vehicle v : vehicleList) {
            Rentable r = (Rentable) v;
            if (r.isAvailable()) {
                available++;
            } else {
                rented++;
                // 只累加已租车辆的收入
                totalRevenue += r.calculateRent(rentDays);
            }
        }

        String info = String.format(
                "📊 租赁统计\n" +
                        "可租车辆：%d 辆\n" +
                        "已租车辆：%d 辆\n" +
                        "💰 预计总收入：%.1f 元",
                available, rented, totalRevenue
        );

        statsLabel.setText(info);
    }

    private void showAlert(String msg) {
        Alert alert = new Alert(Alert.AlertType.WARNING);
        alert.setTitle("提示");
        alert.setContentText(msg);
        alert.show();
    }

    private void showInfo(String title, String msg) {
        Alert alert = new Alert(Alert.AlertType.INFORMATION);
        alert.setTitle(title);
        alert.setContentText(msg);
        alert.show();
    }
}