mod events;

use std::thread::sleep;
use std::time::{Duration, SystemTime, UNIX_EPOCH};



// note(nasr): target timestamp is a unix timestamp of the size u64 that will be compared to the
// current unix timestamp and then trigger an event
