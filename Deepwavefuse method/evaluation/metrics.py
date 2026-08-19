import os

import cv2
import numpy as np
import pandas as pd


def entropy(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    hist = cv2.calcHist(
        [gray],
        [0],
        None,
        [256],
        [0,256]
    )

    hist = hist / hist.sum()

    hist = hist[hist > 0]

    return float(
        -np.sum(
            hist * np.log2(hist)
        )
    )


def spatial_frequency(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    ).astype(np.float32)

    rf = np.diff(
        gray,
        axis=0
    )

    cf = np.diff(
        gray,
        axis=1
    )

    rf = np.sqrt(
        np.mean(rf**2)
    )

    cf = np.sqrt(
        np.mean(cf**2)
    )

    return float(
        np.sqrt(
            rf**2 + cf**2
        )
    )


def average_gradient(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    ).astype(np.float32)

    gx = np.gradient(
        gray,
        axis=1
    )

    gy = np.gradient(
        gray,
        axis=0
    )

    return float(
        np.mean(
            np.sqrt(
                gx**2 + gy**2
            )
        )
    )


def main():

    folder = "./results"

    files = sorted([

        f for f in os.listdir(folder)

        if f.lower().endswith(
            (
                ".jpg",
                ".png",
                ".jpeg"
            )
        )

    ])

    table = []

    print("="*60)
    print("DeepWaveFuse Evaluation")
    print("="*60)

    for file in files:

        image = cv2.imread(
            os.path.join(
                folder,
                file
            )
        )

        en = entropy(image)

        sf = spatial_frequency(image)

        ag = average_gradient(image)

        table.append(

            [

                file,

                en,

                sf,

                ag

            ]

        )

        print(
            file,
            "EN:",
            round(en,4),
            "SF:",
            round(sf,4),
            "AG:",
            round(ag,4)
        )

    df = pd.DataFrame(

        table,

        columns=[

            "Image",

            "Entropy",

            "SF",

            "AG"

        ]

    )

    os.makedirs(
        "./evaluation",
        exist_ok=True
    )

    csv_path = "./evaluation/metrics.csv"

    df.to_csv(
        csv_path,
        index=False
    )

    print()

    print(df.mean(numeric_only=True))

    print()

    print("Saved :", csv_path)


if __name__ == "__main__":

    main()
