// Car.java
public class Car extends Vehicle implements Rentable {
    private int seats;
    private String fuelType; // "汽油" or "电动"

    public Car(String licensePlate, String brand, int year, double dailyRate,
               int seats, String fuelType) {
        super(licensePlate, brand, year, dailyRate);
        this.seats = seats;
        this.fuelType = fuelType;
    }

    @Override
    public double calculateRent(int days) {
        double baseRent = dailyRate * days;
        if ("电动".equals(fuelType)) {
            return baseRent * 0.9;
        }
        return baseRent;
    }

    @Override
    public double calculateMaintenanceCost() {
        double baseCost = getAge() * 100;
        if (seats > 5) {
            baseCost += 500;
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

    public void displayCarInfo() {
        displayInfo();
        System.out.printf("类型: 轿车, 座位数: %d, 燃油类型: %s%n", seats, fuelType);
    }
}
