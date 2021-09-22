#!/usr/bin/env bash

# password-free sudo
echo "%sudo   ALL=(ALL:ALL) NOPASSWD: ALL" >> /etc/sudoers

# create user
groupadd -g $gid $group
useradd -u $uid -g $gid -s /bin/bash -d $home -G sudo $user

# switch user
exec su $user
