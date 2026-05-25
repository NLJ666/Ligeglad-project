// Vehicle.java
import java.time.Year;

// VehicleRentalSystem.java
public class VehicleRentalSystem {
    public static void main(String[] args) {
        System.out.println("所有车辆信息：");

        // 创建4辆不同的车辆
        Vehicle[] vehicles = new Vehicle[4];

        // 1. 轿车 - Toyota
        vehicles[0] = new Car("京A12345", "Toyota", 2022, 300.0, 5, "汽油");

        // 2. 轿车 - Tesla
        vehicles[1] = new Car("京B67890", "Tesla", 2023, 400.0, 5, "电动");

        // 3. 货车 - 解放
        vehicles[2] = new Truck("沪C54321", "解放", 2020, 800.0, 3.5, true);

        // 4. 客车 - 宇通
        vehicles[3] = new Bus("粤D98765", "宇通", 2021, 1200.0, 45, true);

        // 显示所有车辆信息
        for (Vehicle vehicle : vehicles) {
            if (vehicle instanceof Car) {
                ((Car) vehicle).displayCarInfo();
            } else if (vehicle instanceof Truck) {
                ((Truck) vehicle).displayTruckInfo();
            } else if (vehicle instanceof Bus) {
                ((Bus) vehicle).displayBusInfo();
            }
            System.out.println();
        }

        System.out.println("出租操作：");
        // 出租2辆车
        Rentable car1 = (Rentable) vehicles[0];
        Rentable car2 = (Rentable) vehicles[1];
        car1.rent("客户1");
        car2.rent("客户2");
        System.out.println();

        // 归还1辆车
        System.out.println("归还操作：");
        car1.returnVehicle();
        System.out.println();

        // 计算租金（租5天）
        System.out.println("租金计算（租5天）：");
        int rentDays = 5;
        double totalRevenue = 0;

        for (Vehicle vehicle : vehicles) {
            Rentable rentable = (Rentable) vehicle;
            double rent = rentable.calculateRent(rentDays);
            double maintenanceCost = vehicle.calculateMaintenanceCost();
            totalRevenue += rent;

            System.out.printf("%s - 租金: %.1f元, 维护费: %.1f元%n",
                    vehicle.getLicensePlate(), rent, maintenanceCost);
        }
        System.out.println();

        // 统计信息
        int availableCount = 0;
        int rentedCount = 0;

        for (Vehicle vehicle : vehicles) {
            Rentable rentable = (Rentable) vehicle;
            if (rentable.isAvailable()) {
                availableCount++;
            } else {
                rentedCount++;
            }
        }

        System.out.println("统计信息：");
        System.out.println("可租车辆数量: " + availableCount);
        System.out.println("已租车辆数量: " + rentedCount);
        System.out.println("预计总收入: " + totalRevenue + "元");
    }
}