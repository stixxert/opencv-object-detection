#pragma once

#include <chrono>

class timer {

public:

	// Create and start a new Timer
	timer();

	// Copy construction and assignment
	timer(const timer&) = default;
	~timer() = default;

	// Destructor
	timer& operator=(const timer&) = default;

	// Reset the Timer start to now
	void reset();

	// Return the elapsed time since starting (or last reset) in seconds
	// Time is given to millisecond precision (3 decimal places)
	double elapsed() const;

private:

	std::chrono::time_point<std::chrono::high_resolution_clock> start;
};
