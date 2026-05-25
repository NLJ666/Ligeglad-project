// Bus.java
public class Bus extends Vehicle implements Rentable {
    private int passengerCapacity;
    private boolean hasTV;

    public Bus(String licensePlate, String brand, int year, double dailyRate,
               int passengerCapacity, boolean hasTV) {
        super(licensePlate, brand, year, dailyRate);
        this.passengerCapacity = passengerCapacity;
        this.hasTV = hasTV;
    }

    @Override
    public double calculateRent(int days) {
        double baseRent = dailyRate * days;
        double totalRent = baseRent;

        // 载客量 > 20人，加收30%租金
        if (passengerCapacity > 20) {
            totalRent += baseRent * 0.3;
        }

        // 有电视额外加收10%租金
        if (hasTV) {
            totalRent += baseRent * 0.1;
        }

        return totalRent;
    }

    @Override
    public double calculateMaintenanceCost() {
        double baseCost = getAge() * 150;
        baseCost += passengerCapacity * 20;
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

    public void displayBusInfo() {
        displayInfo();
        System.out.printf("类型: 客车, 载客: %d人, 电视: %s%n",
                passengerCapacity, hasTV ? "是" : "否");
    }
}
