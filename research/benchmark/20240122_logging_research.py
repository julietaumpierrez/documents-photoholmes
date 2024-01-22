# %%
import logging

# Define a new log level
VERBOSE = 15
logging.addLevelName(VERBOSE, "VERBOSE")


# Create a new function for the new log level
def verbose(self, message, *args, **kws):
    if self.isEnabledFor(VERBOSE):
        self._log(VERBOSE, message, args, **kws)


# %%

# Add the new function to the logger
logging.Logger.verbose = verbose

# # Create a logger
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Use the new log level
logger.setLevel(VERBOSE)
# %%
logger.verbose("This is a verbose message.")

# Use the info log level
# %%
logger.setLevel(logging.INFO)
logger.info("This is an info message.")

# %%
logger.verbose("This is a verbose message.")

# %%
print(logging._levelToName[15])

# %%
