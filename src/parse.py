import matplotlib.pyplot as plt

nums = []
for eachline in open('/Users/marcuslinture/Desktop/SuperMarioBrosRL/src/output.log', 'r'):
    if eachline.startswith('episode'):
        nums.append(float(eachline[len(eachline)-7:len(eachline)-1]))

episodes = [i for i in range(len(nums))]

plt.plot(episodes, nums)
plt.ylim(bottom=0)
plt.show()

