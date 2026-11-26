from .main import main
import sys

if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print(
            "\nInterrupted by user.",
            file=sys.stderr,
        )

    except Exception as error:
        print(
            f"Fatal error: {error}",
            file=sys.stderr,
        )
        sys.exit(1)
