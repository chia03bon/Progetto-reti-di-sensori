from classes.Thingy52Client_test import Thingy52Client
from utils.utility import scan, find
import asyncio


async def main():
    my_thingy_addresses = ["FD:CD:F2:2A:C9:6A"]
    discovered_devices = await scan()
    my_devices = find(discovered_devices, my_thingy_addresses)

    thingy52 = Thingy52Client(my_devices[0])
    await thingy52.connect()
    thingy52.save_to(str(input("Press enter to test data...")))#test
    await thingy52.receive_inertial_data()


if __name__ == '__main__':
    asyncio.run(main())