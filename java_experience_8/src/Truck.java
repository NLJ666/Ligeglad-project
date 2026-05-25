// Truck.java
public class Truck extends Vehicle implements Rentable {
    private double loadCapacity;
    private boolean hasRefrigeration;

    public Truck(String licensePlate, String brand, int year, double dailyRate,
                 double loadCapacity, boolean hasRefrigeration) {
        super(licensePlate, brand, year, dailyRate);
        this.loadCapacity = loadCapacity;
        this.hasRefrigeration = hasRefrigeration;
    }

    @Override
    public double calculateRent(int days) {
        // 计算每天的基本租金费率
        double dailyChargeRate = 1.0; // 基础费率

        // 载重 > 2吨，每多0.5吨每天租金加收10%
        if (loadCapacity > 2.0) {
            double extraTons = loadCapacity - 2.0;
            int additionalCharges = (int) Math.ceil(extraTons / 0.5);
            dailyChargeRate += 0.1 * additionalCharges;
        }

        // 冷藏功能每天租金额外加收20%
        if (hasRefrigeration) {
            dailyChargeRate += 0.2;
        }

        // 总租金 = 每日租金 × 费率 × 天数
        return dailyRate * dailyChargeRate * days;
    }

    @Override
    public double calculateMaintenanceCost() {
        double baseCost = getAge() * 200;
        if (hasRefrigeration) {
            baseCost += 1000;
        }
        return baseCost;
    }

    @Override
    public boolean isAvailable() {
        return !isRented;
    }

    @Override
    public void rent(String customerName) {
        if (!isRented) {
            isRented = true;
            currentRenter = customerName;
            System.out.println(licensePlate + " 已出租给" + customerName);
        } else {
            System.out.println(licensePlate + " 当前不可租");
        }
    }

    @Override
    public void returnVehicle() {
        if (isRented) {
            isRented = false;
            currentRenter = null;
            System.out.println(licensePlate + " 已归还");
        } else {
            System.out.println(licensePlate + " 当前未被租用");
        }
    }

    public void displayTruckInfo() {
        displayInfo();
        System.out.printf("类型: 货车, 载重: %.1f吨, 冷藏: %s%n",
                loadCapacity, hasRefrigeration ? "是" : "否");
    }
}
