import math


class KLScheduler:
    def __init__(self, start=0.0, stop=1.0, n_samples=int(1e6)):
        self.start = start
        self.stop = stop
        self.n_samples = n_samples
        self.num_samples_processed = 0

        self.B = math.log(start)
        self.A = (math.log(stop) - self.B) / n_samples

    def step(self, n):
        "n: the number of data points in the step"
        self.num_samples_processed += n

    def get_kl_weight(self):
        return min(math.exp(self.A * self.num_samples_processed + self.B), self.stop)
    
    def __repr__(self):
        return f"KLScheduler(start={self.start}, stop={self.stop}, n_samples={self.n_samples})"
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    kl_scheduler = KLScheduler(1e-8, 1, 100)
    
    weights = []
    for i in range(120):
        kl_scheduler.step(1)
        weights.append(kl_scheduler.get_kl_weight())

    plt.plot(weights, 'k-')
    plt.axhline(1e-8, color="red", linestyle="--")
    plt.axhline(1, color="red", linestyle="--")

    plt.savefig("kl_scheduler.png")
    
