
import matplotlib.pyplot as plt
xx = [1,2,3,4]
x = [1,2,3,4]

# y = [173.25, 343.57, 517.82, 694.65 ]
# y = [56.5, 110.3, 167.1, 220.4]
y = [157.5, 312.3, 465.1, 618.3]
plt.figure(figsize=(10, 8))
plt.plot(x,y,"ks-")
for x, y in zip(x, y):
        plt.text(x, y+6, str(y), ha='center', va='bottom', fontsize=18)

plt.xlabel('Node Number', fontname='Arial', fontsize=22)
plt.ylabel('Edges Generation Speed(edges/s)   x10000', fontname='Arial', fontsize=22)
# plt.legend(loc='upper right', fontsize=22)
plt.tick_params(labelsize=18)
plt.xticks(xx)
plt.savefig('grapyline.jpg', bbox_inches='tight')
plt.show()