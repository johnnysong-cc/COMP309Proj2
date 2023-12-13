import base64
from io import BytesIO

import numpy as np
from matplotlib import pyplot as plt


def create_plot(data, title):
    if not data:
        raise ValueError(f"No data provided for {title}")

    fig, ax = plt.subplots()
    ax.bar(data.keys(), data.values())
    ax.set_title(title)
    ax.set_ylabel('Values')
    ax.set_xticklabels(data.keys(), rotation=45, ha='right')

    # Save the plot to a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close(fig)
    buffer.seek(0)

    # Encode the image as base64 and decode it to a string
    image_png = buffer.getvalue()
    image_b64 = base64.b64encode(image_png).decode()

    return f"data:image/png;base64,{image_b64}"


def getstats(df):
    numeric_df = df.select_dtypes(include=[np.number])

    stats = {
        'mean': numeric_df.mean().to_dict(),
        'median': numeric_df.median().to_dict(),
        'standard_deviation': numeric_df.std().to_dict()
    }

    return stats
