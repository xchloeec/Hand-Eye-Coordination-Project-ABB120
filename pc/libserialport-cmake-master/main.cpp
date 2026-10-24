/*********************************************************************
Robot Communication code provided to students in RME40003 Robot System
Design. This code uses the libserialport to commuincate through a 
serial port to the I/O interface box, which in turn links to the 
robot.

To get started students are advised to change the value of cmd in line 
55 and build and run as per the instructions in the Robot Quick 
Start Guides provided. The value of cmd will be sent to the I/O 
interface box and can be seen by the lights indicated on the box.

This code was developed by Michelle Dunn and Scott Fraser, Swinburne
University of Technology, September 2019
*********************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <libserialport.h>

#define BAUD 9600

int main(int argc, char *argv[])
{
	struct sp_port *port;
	int err;
	int key = 0;
	char cmd;

	/* Set up and open the port */
	/* check port usage */
	if (argc < 2)
	{
		/* return error */
		fprintf(stderr, " Port use\n");
		exit(1);
	}

	/* get port name */
	err = sp_get_port_by_name(argv[1], &port);
	if (err == SP_OK)
		/* open port */
		err = sp_open(port, SP_MODE_WRITE);
	if (err != SP_OK)
	{
		/* return error */
		fprintf(stderr, " Can't open port %s\n", argv[1]);
		exit(2);
	}

	/* set Baud rate */
	sp_set_baudrate(port, BAUD);
	/* set the number of bits */
	sp_set_bits(port, 8);

	/* specify the comand to send to the port */ 
	cmd = 255;

	/* set up to exit when q key is entered */
	while (key != 'q') {
		/* write the number "cmd" to the port */
		sp_blocking_write(port, &cmd, 1, 100);

		/*check if a key has been entered */
		key = getchar();

	}

	/* close the port */
	sp_close(port);
	return 0;
}