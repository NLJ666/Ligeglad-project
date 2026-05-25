import java.time.Year;

public abstract class Vehicle {
    protected String licensePlate;
    protected String brand;
    protected int year;
    protected double dailyRate;
    protected boolean isRented;
    protected String currentRenter;

    public Vehicle(String licensePlate, String brand, int year, double dailyRate) {
        this.licensePlate = licensePlate;
        this.brand = brand;
        this.year = year;
        this.dailyRate = dailyRate;
        this.isRented = false;
        this.currentRenter = null;
    }

    public abstract double calculateMaintenanceCost();

    public void displayInfo() {
        System.out.printf("车牌号: %s, 品牌: %s, 购买年份: %d, 日租金: %.1f%n",
                licensePlate, brand, year, dailyRate);
    }

    public int getAge() {
        int currentYear = Year.now().getValue();
        return currentYear - year;
    }

    // Getter methods
    public String getLicensePlate() { return licensePlate; }
    public boolean getIsRented() { return isRented; }
    public String getCurrentRenter() { return currentRenter; }
    public double getDailyRate() { return dailyRate; }
}
