import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from typing import Literal, Optional
from datetime import datetime
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter


def format_thousands(x, pos):
    """Formatter function to add commas as a thousand separators."""
    return f'{int(x):,}'.replace(',', ' ')


def spectrum_above_mask(spectrum: np.array,
                        mask: np.array = None,
                        frequency_bins: np.array = None,
                        time_bins: np.array = None,
                        sample_id='unknown',
                        output_mode: Optional[Literal['display', 'return']] = 'display'):
    """
    Plots the spectrum and optionally a mask, with frequency and time bin annotations.

    :param spectrum: 2D NumPy array representing the spectrogram.
    :param mask: 2D NumPy array representing the segmentation mask (optional).
    :param frequency_bins: 1D NumPy array of frequency bin labels (optional).
    :param time_bins: 1D NumPy array of time bin labels (optional).
    :param sample_id: Identifier for the sample being plotted.
    :param output_mode: Specifies the output mode. Acceptable values:
                          - 'display': Show the plot interactively and return it as an PIL image.
                          - 'save': Save and return the plot as a PIL image object.
                          - None: Return plot as PIL image.
    :return: None or a PIL image, depending on return_format.
    """

    # Determine the number of subplots based on the presence of the mask
    n_rows = 2 if mask is not None else 1
    fig, ax = plt.subplots(n_rows, 1, figsize=(12, 8))

    if n_rows == 1:
        ax = [ax]  # Ensure ax is always a list for consistency

    # Displaying the spectrogram (spectrum)
    cax1 = ax[0].imshow(spectrum, aspect='auto', origin='lower', cmap='inferno')
    ax[0].set_title(f'Spectrogram of sample ID_{sample_id}')

    # Set x and y axis ticks
    n_time_ticks = 5
    n_freq_ticks = 5

    # Set ticks for time axis
    if frequency_bins is not None and time_bins is not None:
        time_tick_indices = np.linspace(0, len(time_bins) - 1, n_time_ticks, dtype=int)
        time_tick_labels = [f'{time_bins[i]:.2f}' for i in time_tick_indices]
        ax[0].set_xticks(time_tick_indices)
        ax[0].set_xticklabels(time_tick_labels)

        # Name axis
        ax[0].set_ylabel('Frequency [Hz]')
        ax[0].set_xlabel('Time [s]')
    else:
        ax[0].set_ylabel('Spectrum [bins]')
        ax[0].set_xlabel('Time [bins]')

    # Set ticks for frequency axis
    if frequency_bins is not None and time_bins is not None:
        freq_tick_indices = np.linspace(0, len(frequency_bins) - 1, n_freq_ticks, dtype=int)
        freq_tick_labels = [f'{frequency_bins[i]:.2f}' for i in freq_tick_indices]
        ax[0].set_yticks(freq_tick_indices)
        ax[0].set_yticklabels(freq_tick_labels)

    # Displaying the mask if it exists
    if mask is not None:
        cax2 = ax[1].imshow(mask, aspect='auto', origin='lower', cmap='gray')
        ax[1].set_title(f'Segmentation Mask for sample ID_{sample_id}')

        # Set x and y axis ticks for the mask
        if frequency_bins is not None and time_bins is not None:
            ax[1].set_xticks(time_tick_indices)
            ax[1].set_xticklabels(time_tick_labels)
            ax[1].set_yticks(freq_tick_indices)
            ax[1].set_yticklabels(freq_tick_labels)

            # Name axis
            ax[1].set_ylabel('Frequency [Hz]')
            ax[1].set_xlabel('Time [s]')
        else:
            ax[1].set_ylabel('Frequency [bins]')
            ax[1].set_xlabel('Time [bins]')

    # Adjust layout
    plt.tight_layout()

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    Image_PIL = Image.fromarray(image)

    # Show plots
    if output_mode == 'display':
        plt.show()

    if output_mode == 'save':
        # # Save the plot to a BytesIO object
        # buf = BytesIO()
        # plt.savefig(buf, format='png')
        # buf.seek(0)
        #
        # # Create a PIL image from the buffer
        # img = Image.open(buf)
        # buf.close()

        current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        Image_PIL.save(f'{current_datetime}_{sample_id}.jpg')

    # Return the image
    return Image_PIL


def plot_spectrogram_with_segments(time_bins, freq_bins, labels, spectrum, title=None, xlim=None):
    """
    Plot spectrogram with annotations.
    :param time_bins: Time bins (array of time values)
    :param freq_bins: Frequency bins (array of frequency values)
    :param labels: Labels for annotations (array of label values)
    :param spectrum: Spectrogram (2D array of intensities)
    :param title: Title of the plot (optional)
    :param xlim: x-axis limits (tuple, optional)
    :return: None
    """
    # Create a mapping from label to color
    label_colors = [
        'red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta',
        'yellow', 'brown', 'pink', 'gray', 'lime', 'teal', 'navy',
        'coral', 'gold', 'salmon', 'violet', 'khaki', 'lightblue'
    ]
    unique_labels = np.unique(labels)
    if len(unique_labels) > len(label_colors):
        print(
            f"Warning: More unique labels ({len(unique_labels)}) than available colors ({len(label_colors)}). Defaulting to gray.")
        label_color_map = {label: 'gray' for label in unique_labels}
    else:
        label_color_map = {label: label_colors[i % len(label_colors)] for i, label in
                           enumerate(unique_labels)}

    # Plot the spectrogram
    plt.figure(figsize=(12, 6))
    plt.imshow(spectrum, aspect='auto', origin='lower',
               extent=[time_bins[0], time_bins[-1], freq_bins[0], freq_bins[-1]], cmap='viridis')
    plt.colorbar(label='Intensity [dB]')
    if title is not None:
        plt.title(title, y=1.10)
    else:
        plt.title('Spectrogram with Annotations', y=1.05)
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')

    # Format y-axis ticks with thousands separators
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_thousands))

    # Add annotations to the spectrogram
    current_label = None
    start_time = None

    for i, label in enumerate(labels):
        if label != 0:  # If the label is not zero, it's a valid label
            if current_label is None:  # Starting a new label
                current_label = label
                start_time = time_bins[i]
            elif current_label != label:  # Switching to a new label
                end_time = time_bins[i]
                # Draw a colored rectangle for the current label
                plt.gca().add_patch(
                    Rectangle((start_time, freq_bins[0]), end_time - start_time, freq_bins[-1] - freq_bins[0],
                              linewidth=0, facecolor=label_color_map[current_label], alpha=0.3)
                )
                # Draw a border for the current label
                plt.gca().add_patch(
                    Rectangle((start_time, freq_bins[0]), end_time - start_time, freq_bins[-1] - freq_bins[0],
                              linewidth=1, edgecolor='red', facecolor='none', linestyle='--', alpha=0.7)
                )
                # Add text with the label name
                plt.text((start_time + end_time) / 2, freq_bins[-1] * 1.02, str(int(current_label)),
                         fontsize=10, color='red', ha='center', va='bottom')
                current_label = label
                start_time = time_bins[i]
        elif current_label is not None:  # End of the current label sequence
            end_time = time_bins[i]
            plt.gca().add_patch(
                Rectangle((start_time, freq_bins[0]), end_time - start_time, freq_bins[-1] - freq_bins[0],
                          linewidth=0, facecolor=label_color_map[current_label], alpha=0.3)
            )
            plt.gca().add_patch(
                Rectangle((start_time, freq_bins[0]), end_time - start_time, freq_bins[-1] - freq_bins[0],
                          linewidth=1, edgecolor='red', facecolor='none', linestyle='--', alpha=0.7)
            )
            plt.text((start_time + end_time) / 2, freq_bins[-1] * 1.02, str(int(current_label)),
                     fontsize=10, color='red', ha='center', va='bottom')
            current_label = None
            start_time = None

    # Ensure the last label is printed if it exists
    if current_label is not None:
        plt.gca().add_patch(
            Rectangle((start_time, freq_bins[0]), time_bins[-1] - start_time, freq_bins[-1] - freq_bins[0],
                      linewidth=0, facecolor=label_color_map[current_label], alpha=0.3)
        )
        plt.gca().add_patch(
            Rectangle((start_time, freq_bins[0]), time_bins[-1] - start_time, freq_bins[-1] - freq_bins[0],
                      linewidth=1, edgecolor='red', facecolor='none', linestyle='--', alpha=0.7)
        )
        plt.text((start_time + time_bins[-1]) / 2, freq_bins[-1] * 1.02, str(int(current_label)),
                 fontsize=10, color='red', ha='center', va='bottom')

    # If xlim is provided, set the limits of the x-axis
    if xlim is not None:
        plt.xlim(*xlim)  # Use the tuple (xmin, xmax) provided by the user

        # Adjust the positions of the text annotations so they respect the xlim
        for text in plt.gca().texts:
            # Adjust the position of each text (based on xlim)
            text_x = text.get_position()[0]
            if text_x < xlim[0] or text_x > xlim[1]:
                text.set_visible(False)

    # Show the plot
    plt.tight_layout()
    plt.show()
