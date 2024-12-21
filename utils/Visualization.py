import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from io import BytesIO
from PIL import Image


class VisualizationHandler:
    def __init__(self, label_colors=None):
        # Initialize the class with optional label colors
        if label_colors is None:
            # Default color set if none is provided
            self.label_colors = [
                'red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta',
                'yellow', 'brown', 'pink', 'gray', 'lime', 'teal', 'navy',
                'coral', 'gold', 'salmon', 'violet', 'khaki', 'lightblue'
            ]
        else:
            self.label_colors = label_colors

    def format_thousands(self, x, pos):
        """Formatter function to add commas as a thousand separators."""
        return f'{int(x):,}'.replace(',', ' ')

    def plot_spectrogram_with_annotations(self, time_bins, freq_bins, labels, spectrum, title=None, xlim=None):
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
        unique_labels = np.unique(labels)
        if len(unique_labels) > len(self.label_colors):
            print(
                f"Warning: More unique labels ({len(unique_labels)}) than available colors ({len(self.label_colors)}). Defaulting to gray.")
            label_color_map = {label: 'gray' for label in unique_labels}
        else:
            label_color_map = {label: self.label_colors[i % len(self.label_colors)] for i, label in
                               enumerate(unique_labels)}

        # Plot the spectrogram
        plt.figure(figsize=(12, 6))
        plt.imshow(spectrum, aspect='auto', origin='lower',
                   extent=[time_bins[0], time_bins[-1], freq_bins[0], freq_bins[-1]], cmap='viridis')
        plt.colorbar(label='Intensity (dB)')
        if title is not None:
            plt.title(title, y=1.10)
        else:
            plt.title('Spectrogram with Annotations', y=1.05)
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')

        # Format y-axis ticks with thousands separators
        plt.gca().yaxis.set_major_formatter(FuncFormatter(self.format_thousands))

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

    def plot_spectrogram(self, time_bins, freq_bins, spectrum, title=None):
        """
        Plot spectogram without annotations.
        """
        extent = [time_bins[0], time_bins[-1], freq_bins[0], freq_bins[-1]] if freq_bins is not None and time_bins is not None else None
        plt.figure(figsize=(12, 6))
        plt.imshow(spectrum, aspect='auto', origin='lower', extent=extent, cmap='viridis')
        plt.colorbar(label='Intensity (dB)')

        if title is not None:
            plt.title(title, y=1.10)
        else:
            plt.title('Spectrogram', y=1.05)

        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')

        # Format y-axis ticks with thousands separators
        plt.gca().yaxis.set_major_formatter(FuncFormatter(self.format_thousands))
        plt.tight_layout()

        # Uložení grafu do bufferu
        buf = BytesIO()
        plt.savefig(buf, format='png')  # Uloží do bufferu jako PNG
        plt.close()  # Zavře graf, aby nezahlcoval paměť

        # Přesun ukazatele bufferu na začátek
        buf.seek(0)

        # Načtení obrazu z bufferu
        image = Image.open(buf).copy()
        buf.close()

        return image

    def plot_spectrogram_with_segments(self, time_bins, freq_bins, spectrum, segments, title=None):
        """
        Plotting spectogram with only segments highlighted (using vertical lines).
        """
        plt.figure(figsize=(12, 6))
        plt.imshow(spectrum, aspect='auto', origin='lower',
                   extent=[time_bins[0], time_bins[-1], freq_bins[0], freq_bins[-1]], cmap='viridis')
        plt.colorbar(label='Intensity (dB)')

        if title is not None:
            plt.title(title, y=1.10)
        else:
            plt.title('Spectrogram with Segment Separators', y=1.05)

        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')

        # Format y-axis ticks with thousands separators
        plt.gca().yaxis.set_major_formatter(FuncFormatter(self.format_thousands))

        # Add vertical lines where segments are 1
        for i, segment in enumerate(segments):
            if segment == 1:
                plt.axvline(x=time_bins[i], color='red', linestyle='--', linewidth=1)

        # Show the plot
        plt.tight_layout()
        plt.show()
