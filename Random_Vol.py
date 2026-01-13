import random
import matplotlib.pyplot as plt
import numpy as np

class SawtoothRandomGenerator:
    
    def __init__(self, min_bound=0, max_bound=5, max_step=0.01):
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.range_size = max_bound - min_bound
        self.max_step = max_step
        
    def generate_sawtooth_series(self, num_points=30000, reset_randomness=True):
        data = []
        current_value = self.min_bound
        
        for i in range(num_points):
            step = random.uniform(0, self.max_step)
            
            if current_value + step > self.max_bound:
                current_value = self.min_bound
                if reset_randomness:
                    current_value += random.uniform(0, self.max_step * 2)
            else:
                current_value += step
            
            current_value = max(self.min_bound, min(self.max_bound, current_value))
            data.append(current_value)
        
        return data
    
def visualize_sawtooth_patterns():
    plt.figure(figsize=(16, 9))
    
    generator = SawtoothRandomGenerator(min_bound=0.2, max_bound=4.8, max_step=0.01)
    data = generator.generate_sawtooth_series(num_points=5000)
    
    plt.subplot(3, 1, 1)
    plt.plot(data, color='blue')
    plt.title('Voltage data simulation of enforced running', fontsize=14, fontweight='bold')
    plt.xlabel('Frame', fontsize=12)
    plt.ylabel('Voltage (V)', fontsize=12)
    plt.grid(False)

    delta_voltage = 5
    thresh = 3 / 5 * delta_voltage
    calibration_data = np.array(data)
    diff_data = np.diff(calibration_data)
    ind = np.where(np.abs(diff_data) > thresh)[0]
    for i in ind:
        if diff_data[i] < -thresh:
            calibration_data[i + 1:] += delta_voltage
        elif diff_data[i] > thresh:
            calibration_data[i + 1:] -= delta_voltage
    
    plt.subplot(3, 1, 2)
    plt.plot(np.diff(calibration_data), color='red')
    plt.title('ΔV simulation of enforced running after original calibration method', fontsize=14, fontweight='bold')
    plt.xlabel('Frame', fontsize=12)
    plt.ylabel('ΔV', fontsize=12)
    plt.grid(False)

    calibration_data1 = np.array(data)
    diff_data = np.diff(calibration_data1)
    ind = np.where(np.abs(diff_data) > thresh)[0]
    for i in ind:
        if diff_data[i] < -thresh:
            calibration_data1[i + 1:] += (abs(diff_data[i]) + (diff_data[i-1]+diff_data[i+1])/2)
        elif diff_data[i] > thresh:
            calibration_data1[i + 1:] -= (abs(diff_data[i]) + (diff_data[i-1]+diff_data[i+1])/2)

    plt.subplot(3, 1, 3)
    plt.plot(np.diff(calibration_data1), color='green')
    plt.title('ΔV simulation of enforced running after new calibration method', fontsize=14, fontweight='bold')
    plt.xlabel('Frame', fontsize=12)
    plt.ylabel('ΔV', fontsize=12)
    plt.grid(False)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_sawtooth_patterns()