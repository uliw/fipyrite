#!/usr/bin/env python3
import time
import argparse
import pandas as pd
import pathlib as pl
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Import project-specific modules
import mantis.plot_data_new as plot_data_new

class PlotUpdateHandler(FileSystemEventHandler):
    """Handles file system events for the results CSV."""

    def __init__(
        self, csv_file, layout_file, display_length, measured_data, output_plot=None
    ):
        super().__init__()
        self.csv_file = csv_file
        self.layout_file = layout_file
        self.display_length = display_length
        self.measured_data = measured_data
        self.output_plot = output_plot
        self.last_trigger_time = 0
        self.debounce_seconds = 1.0

    def on_modified(self, event):
        self._handle_event(event)

    def on_created(self, event):
        self._handle_event(event)

    def on_moved(self, event):
        self._handle_event(event)

    def _handle_event(self, event):
        if event.is_directory:
            return

        # Target files
        target_csv = pl.Path(self.csv_file).resolve()
        target_layout = pl.Path(self.layout_file).resolve()
        
        # Source path of the event
        src_path = pl.Path(event.src_path).resolve()
        # For MoveEvent, we also check dest_path
        dest_path = pl.Path(event.dest_path).resolve() if hasattr(event, 'dest_path') else None

        is_match = (src_path == target_csv or src_path == target_layout or 
                    (dest_path and (dest_path == target_csv or dest_path == target_layout)))

        if is_match:
            current_time = time.time()
            dt = current_time - self.last_trigger_time
            if dt > self.debounce_seconds:
                print(f"[Monitor] DEBUG: Triggered by {event.event_type} on {src_path} (dt={dt:.3f}s)")
                self.last_trigger_time = current_time
                self.trigger_plot()
            else:
                # print(f"[Monitor] DEBUG: Debounced {event.event_type} on {src_path} (dt={dt:.3f}s)")
                pass

    def trigger_plot(self):
        print(f"[Monitor] Change detected. Updating plots...")
        try:
            # Short sleep to ensure the file is fully written/closed by the simulation
            time.sleep(0.5)

            # Load the fresh data
            df = pd.read_csv(self.csv_file)

            # Use specified output name or derive from CSV
            output_path = (
                pl.Path(self.output_plot)
                if self.output_plot
                else pl.Path(self.csv_file).with_suffix(".pdf")
            )

            # Load layout (this allows updates to plot_layout.py to be picked up too)
            plt_desc = plot_data_new.load_layout_from_file(df, self.layout_file, self.measured_data)

            # Execute the plotting function as requested
            plot_data_new.plot(
                df,
                self.display_length,
                output_path,
                show=False,
                plot_description=plt_desc,
                measured_data_path=self.measured_data,
            )
            print(f"[Monitor] Plots updated: {output_path.name}")

        except Exception as e:
            print(f"[Monitor] Error during plotting: {e}")


def main():
    import sys

    parser = argparse.ArgumentParser(
        description="Monitor a CSV file and update plots automatically."
    )
    parser.add_argument(
        "-f", "--file", default="pyrite_model_fipy.csv", help="CSV file to monitor"
    )
    parser.add_argument(
        "-l", "--layout", default="plot_layout.py", help="Plot layout file"
    )
    parser.add_argument(
        "-d",
        "--display_length",
        type=float,
        default=2.0,
        help="Display length in meters",
    )
    parser.add_argument("-m", "--measured", default=None, help="Measured data CSV file")
    parser.add_argument(
        "-o", "--output", help="Output plot filename (default: derived from --file)"
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    # Determine output path for the initial check
    output_path = (
        pl.Path(args.output) if args.output else pl.Path(args.file).with_suffix(".pdf")
    )

    # Setup the observer to watch the current directory
    path = "."
    event_handler = PlotUpdateHandler(
        csv_file=args.file,
        layout_file=args.layout,
        display_length=args.display_length,
        measured_data=args.measured,
        output_plot=args.output,
    )

    # --- Initial Check ---
    if not output_path.exists():
        print(
            f"[Monitor] Output file {output_path.name} not found. Generating initial plot..."
        )
        event_handler.trigger_plot()

    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)

    print(f"Starting Plot Monitor...")
    print(f"Watching: {args.file}")
    if args.output:
        print(f"Output: {args.output}")
    print(f"Layout: {args.layout}")
    print(f"Display Length: {args.display_length}m")
    print(f"Press Ctrl+C to stop.")

    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping monitor...")
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
