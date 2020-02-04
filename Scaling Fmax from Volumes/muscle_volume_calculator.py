import numpy as np

def muscle_volume_calculator(subject_height: float, subject_mass: float):
	'''
	This function calculates subject's muscle volume based on its height (m) and mass (kg).

	'''

	''' Raw Data from Handsfield (2004) '''
	name = ['gluteus maximus', 'adductor magnus', 'gluteus medius', 'psoas', 'iliacus',
    		'sartorius', 'adductor longus', 'gluteus minimus', 'adductor brevis', 'gracilis',
    		'pectineus', 'tensor fasciae latae', 'obturator externus', 'piriformis',
    		'quadratus femoris', 'obturator internus', 'small ext. rotators', 'vastus lateralis',
    		'vastus medialis', 'vastus intermedius', 'rectus femoris', 'semimembranosus',
    		'biceps femoris: l.h.', 'semitendinosus', 'biceps femoris: s.h.', 'popliteus',
    		'soleus', 'med gastrocnemius', 'lat gastrocnemius', 'tibialis anterior',
    		'peroneals (brev/long)', 'tibialis posterior', 'extensors (EDL/EHL)',
    		'flexor hallucis longus', 'flexor digit. longus']

	osim_abbr = ['glut_max', 'add_mag', 'glut_med', 'psoas', 'iliacus',
    			'sar', 'add_long', 'glut_min', 'add_brev', 'grac',
    			'pect', 'tfl', '', 'peri',
    			'quad_fem', '', '', 'vas_lat',
    			'vas_med', 'vas_int', 'rect_fem', 'semimem',
    			'bifemlh', 'semiten', 'bifemsh', '',
    			'soleus', 'med_gas', 'lat_gas', 'tib_ant',
				'per_', 'tib_post', 'ext_',
				'flex_hal', 'flex_dig']

	b1 = np.array([0.123, 0.0793, 0.0478, 0.055, 0.0248, 0.0256, 0.0259, 0.0129, 0.0137, 0.0138, 0.0107, 0.0136, 0.00349,
    	0.00372, 0.00475, 0.00252, 0.00172, 0.125, 0.0631, 0.0273, 0.0371, 0.0319, 0.0256, 0.0285, 0.016, 0.00298,
    	0.0507, 0.0348, 0.0199, 0.0161, 0.0194, 0.0104, 0.0132, 0.0137, 0.00259])

	b2 = np.array([-25.4, -4.7, -16.9, -117, 0.383, -18.2, -21.9, 12.6, 6.2, 6.07, -9.94, -31.5, 28.3, 16.2, -1.45, 8.68,
		3.85, -55.7, -16.3, 76.5, 4.97, 18.2, 24.3, -16.8, -13.8, 2.11, 78.2, 9.42, 8.21, 20.3, -7.43, 30.8,
		8.7, -18.9, 11.6])

	# Currently not used. PCSA and pennation angle are read from the OpenSim model
	# PCSA = np.array([46.8, 45.5, 45.6, 16.5, 12.4, 3.4, 15.4, 8.2, 9.7, 4.7, 5.1, 4, 5.4, 4.7, 4.4, 3.3, 2, 59.3, 59.1,
 	#		39, 34.8, 37.8, 25.9, 9.3, 7.8, 2.5, 124.1, 50.1, 23, 15.8, 19.3, 28.4, 10.2, 16.9, 7.5])

	# penAngle = np.array([21.9, 15.5, 20.5, 10.6, 14.3, 1.3, 7.1, 0, 6.1, 8.2, 0, 0, 0, 0, 0, 0, 0, 18.4, 29.6, 4.5, 0, 
	# 			15.1, 11.6, 12.9, 12.3, 0;28, 3, 9.9, 12, 9.6, 0, 13.7, 10, 16.9, 13.6])

	# Calculate muscle volumes
	V_m_tot = 47 * subject_height * subject_mass + 1285
	muscle_volume = b1 * V_m_tot + b2

	return osim_abbr, muscle_volume
