import sys
sys.coinit_flags = 0

try:
    from bleak.backends.winrt.util import allow_sta
    allow_sta()

except ImportError:
    pass

import asyncio
from utils.utility import scan, connection, find, receive_data
import asyncio
#import multiprocessing
#from utils.plotting import live_plotting

async def main():
    my_thingy_addresses = ("FD:CD:F2:2A:C9:6A")
    discovered_devices = await scan()

    my_devices = find(discovered_devices, my_thingy_addresses)

    connected_thingy_devices = await connection(my_devices)

    input("Press enter to record data...")
    #multiprocessing.Process(target=live_plotting).start()

    await receive_data(connected_thingy_devices)


if __name__ == '__main__':
    asyncio.run(main())