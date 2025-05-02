#!/usr/bin/env python3

import subprocess
from bluezero import peripheral
from pydbus import SystemBus
from gi.repository import GLib

# Storage for incoming Wi-Fi credentials
_wifi_credentials = {'ssid': None, 'password': None}

def ssid_write(value, options):
    _wifi_credentials['ssid'] = value.decode('utf-8')
    _apply_if_ready()

def password_write(value, options):
    _wifi_credentials['password'] = value.decode('utf-8')
    _apply_if_ready()

def _apply_if_ready():
    ssid = _wifi_credentials['ssid']
    pwd  = _wifi_credentials['password']
    if ssid and pwd:
        wpa_conf = f"""ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1
country=US

network={{
    ssid="{ssid}"
    psk="{pwd}"
}}
"""
        with open('/etc/wpa_supplicant/wpa_supplicant.conf', 'w') as f:
            f.write(wpa_conf)
        subprocess.run(['wpa_cli', '-i', 'wlan0', 'reconfigure'])
        print(f'Applied Wi-Fi credentials: SSID="{ssid}"')
        _wifi_credentials['ssid'] = None
        _wifi_credentials['password'] = None

# BLE service and characteristic UUIDs
WIFI_SERVICE_UUID      = '75c4c319-7e6b-46d6-8baf-959dc2651cd6'
SSID_CHAR_UUID         = 'fe8bfaa6-1d17-4442-b106-032966c599cb'
PASSWORD_CHAR_UUID     = '83a14951-e059-4a35-96d8-21aa05910b81'

# Create GATT characteristics
ssid_char = peripheral.Characteristic(
    uuid=SSID_CHAR_UUID,
    properties=['write'],
    write_callback=ssid_write
)
password_char = peripheral.Characteristic(
    uuid=PASSWORD_CHAR_UUID,
    properties=['write'],
    write_callback=password_write
)

# Create GATT service
wifi_service = peripheral.Service(
    uuid=WIFI_SERVICE_UUID,
    characteristics=[ssid_char, password_char]
)

# Build peripheral
ble_periph = peripheral.Peripheral(
    adapter_addr=None,    # auto-select
    local_name='WiFiConfig',
    services=[wifi_service]
)

# DBus PropertiesChanged handler for onConnect / onDisconnect
def on_properties_changed(interface, changed, invalidated, path):
    if interface != 'org.bluez.Device1':
        return
    if 'Connected' in changed:
        if changed['Connected']:
            print(f'Central connected: {path}')
        else:
            print(f'Central disconnected: {path}')

# Setup DBus signal subscription
bus = SystemBus()
bus.subscribe(iface='org.freedesktop.DBus.Properties',
              signal='PropertiesChanged',
              arg0='org.bluez.Device1',
              signal_fired=on_properties_changed)

if __name__ == '__main__':
    # Publish GATT service and start mainloop
    ble_periph.publish()
    print('BLE peripheral advertising as "WiFiConfig"...')
    GLib.MainLoop().run()
