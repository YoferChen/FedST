from glob import glob
import json
import matplotlib.pyplot as plt


def age_counter(image_dir='./aglined faces/'):
    image_list = glob(image_dir + '*')
    print('图像总数：', len(image_list))
    counter = {}
    for image_path in image_list:
        age = image_path.split('A')[-1].split('.')[0]
        if counter.get(age):
            counter[age] += 1
        else:
            counter[age] = 1

    with open("counter.json", 'w', encoding='utf-8') as f:
        json.dump(counter, f, indent=2, ensure_ascii=False)

    x = list(counter.keys())
    x = [int(i) for i in x]
    y = [counter[i] for i in list(counter.keys())]
    plt.bar(x, y)

    for i in range(len(x)):
        plt.text(x[i], y[i], y[i])
    plt.xlabel('Age')
    plt.ylabel('Number')
    plt.show()
