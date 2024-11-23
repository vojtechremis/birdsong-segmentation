import numpy as np
from matplotlib import pyplot as plt
from io import BytesIO
from PIL import Image
from typing import Literal, Optional
from datetime import datetime


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
