from pyimzml.ImzMLWriter import ImzMLWriter
from pyimzml.compression import NoCompression, ZlibCompression
from timsconvert.parse_maldi import *
from functools import partial
import itertools
import multiprocessing
from multiprocessing.pool import ThreadPool as Pool
import numpy as np

def read_maldi_ims_chunk(data, frame_start, frame_stop, mode, exclude_mobility, profile_bins,
                                   encoding):
    if data.meta_data['SchemaType'] == 'TSF':
        list_of_scan_dicts = parse_maldi_tsf(data, frame_start, frame_stop, mode, False, profile_bins, encoding)
    # Parse TDF data.
    elif data.meta_data['SchemaType'] == 'TDF':
        list_of_scan_dicts = parse_maldi_tdf(data, frame_start, frame_stop, mode, False, exclude_mobility, profile_bins,
                                             encoding)
    return list_of_scan_dicts

def write_maldi_ims_chunk_to_imzml1(data, imzml_file, frame_start, frame_stop, mode, exclude_mobility, profile_bins,
                                   encoding, list_of_scan_dicts, rounding=1):
    if data.meta_data['SchemaType'] == 'TSF':
        for scan_dict in list_of_scan_dicts:
            imzml_file.addSpectrum(scan_dict['mz_array'],
                                   scan_dict['intensity_array'],
                                   scan_dict['coord'])
    elif data.meta_data['SchemaType'] == 'TDF':
        if mode == 'profile':
            exclude_mobility = True
        assert rounding >=0, 'mobility_round bits must >= 0'
        if exclude_mobility == False:
             for scan_dict in list_of_scan_dicts:
                all_mobilities = np.round(scan_dict['mobility_array'], decimals=rounding) * 100000 + scan_dict['mz_array']
                imzml_file.addSpectrum(scan_dict['mz_array'],
                                       scan_dict['intensity_array'],
                                       scan_dict['coord'],
                                       mobilities= all_mobilities)
        elif exclude_mobility == True:
            for scan_dict in list_of_scan_dicts:
                imzml_file.addSpectrum(scan_dict['mz_array'],
                                       scan_dict['intensity_array'],
                                       scan_dict['coord'])

def write_maldi_ims_chunk_to_imzml(data, imzml_file, frame_start, frame_stop, mode, exclude_mobility, profile_bins,
                                   encoding):
    # Parse TSF data.
    if data.meta_data['SchemaType'] == 'TSF':
        list_of_scan_dicts = parse_maldi_tsf(data, frame_start, frame_stop, mode, False, profile_bins, encoding)
    # Parse TDF data.
    elif data.meta_data['SchemaType'] == 'TDF':
        list_of_scan_dicts = parse_maldi_tdf(data, frame_start, frame_stop, mode, False, exclude_mobility, profile_bins,
                                             encoding)
    if data.meta_data['SchemaType'] == 'TSF':
        for scan_dict in list_of_scan_dicts:
            imzml_file.addSpectrum(scan_dict['mz_array'],
                                   scan_dict['intensity_array'],
                                   scan_dict['coord'])
    elif data.meta_data['SchemaType'] == 'TDF':
        if mode == 'profile':
            exclude_mobility = True
        if exclude_mobility == False:
            for scan_dict in list_of_scan_dicts:
                imzml_file.addSpectrum(scan_dict['mz_array'],
                                       scan_dict['intensity_array'],
                                       scan_dict['coord'],
                                       mobilities=scan_dict['mobility_array'])
        elif exclude_mobility == True:
            for scan_dict in list_of_scan_dicts:
                imzml_file.addSpectrum(scan_dict['mz_array'],
                                       scan_dict['intensity_array'],
                                       scan_dict['coord'])


def write_maldi_ims_imzml(data, outdir, outfile, mode, exclude_mobility, profile_bins, imzml_mode, encoding,
                          compression, chunk_size, rounding = 1):
    # Set polarity for run in imzML.
    polarity = list(set(data.frames['Polarity'].values.tolist()))
    if len(polarity) == 1:
        polarity = polarity[0]
        if polarity == '+':
            polarity = 'positive'
        elif polarity == '-':
            polarity = 'negative'
        else:
            polarity = None
    else:
        polarity = None

    if data.meta_data['SchemaType'] == 'TSF' and mode == 'raw':
        logging.info(get_timestamp() + ':' + 'TSF file detected. Only export in profile or centroid mode are '
                                             'supported. Defaulting to centroid mode.')

    # Set centroided status.
    if mode == 'profile':
        centroided = False
    elif mode == 'centroid' or mode == 'raw':
        centroided = True

    if encoding == 32:
        encoding_dtype = np.float32
    elif encoding == 64:
        encoding_dtype = np.float64

    # Get compression type object.
    if compression == 'zlib':
        compression_object = ZlibCompression()
    elif compression == 'none':
        compression_object = NoCompression()

    if data.meta_data['SchemaType'] == 'TSF':
        writer = ImzMLWriter(os.path.join(outdir, outfile),
                             polarity=polarity,
                             mode=imzml_mode,
                             spec_type=centroided,
                             mz_dtype=encoding_dtype,
                             intensity_dtype=encoding_dtype,
                             mz_compression=compression_object,
                             intensity_compression=compression_object,
                             include_mobility=False)
    elif data.meta_data['SchemaType'] == 'TDF':
        if mode == 'profile':
            exclude_mobility = True
            logging.info(
                get_timestamp() + ':' + 'Export of ion mobility data is not supported for profile mode data...')
            logging.info(get_timestamp() + ':' + 'Exporting without ion mobility data...')
        if exclude_mobility == False:
            writer = ImzMLWriter(os.path.join(outdir, outfile),
                                 polarity=polarity,
                                 mode=imzml_mode,
                                 spec_type=centroided,
                                 mz_dtype=encoding_dtype,
                                 intensity_dtype=encoding_dtype,
                                 mobility_dtype=encoding_dtype,
                                 mz_compression=compression_object,
                                 intensity_compression=compression_object,
                                 mobility_compression=compression_object,
                                 include_mobility=True)
        elif exclude_mobility == True:
            writer = ImzMLWriter(os.path.join(outdir, outfile),
                                 polarity=polarity,
                                 mode=imzml_mode,
                                 spec_type=centroided,
                                 mz_dtype=encoding_dtype,
                                 intensity_dtype=encoding_dtype,
                                 mz_compression=compression_object,
                                 intensity_compression=compression_object,
                                 include_mobility=False)

    logging.info(get_timestamp() + ':' + 'Writing to .imzML file ' + os.path.join(outdir, outfile) + '...')
    with writer as imzml_file:
        chunk = 0
        frames = list(data.frames['Id'].values)
        while chunk + chunk_size + 1 <= len(frames):
            chunk_list = []
            for i, j in zip(frames[chunk:chunk + chunk_size], frames[chunk + 1: chunk + chunk_size + 1]):
                chunk_list.append((int(i), int(j)))
            logging.info(get_timestamp() + ':' + 'Parsing and writing Frame ' + ':' + str(chunk_list[0][0]) + '...')
            # for frame_start, frame_stop in chunk_list:
            #     write_maldi_ims_chunk_to_imzml(data, imzml_file, frame_start, frame_stop, mode, exclude_mobility,
            #                                    profile_bins, encoding)
            
            frame_start,frame_stop = chunk+1, chunk + chunk_size +1
            list_scan = read_maldi_ims_chunk(data, frame_start, frame_stop, mode, exclude_mobility, profile_bins,
                                   encoding)
            # func = partial(read_maldi_ims_chunk, data, mode=mode, exclude_mobility= exclude_mobility,
            #                                    profile_bins=profile_bins, encoding=encoding)
            # cores = chunk_size #multiprocessing.cpu_count()
            # #chunk_list =  [(d,*fr) for d, fr in zip(dataLs, chunk_list)]
            # with Pool(processes=cores) as pool:
            #     res = pool.starmap(func, chunk_list)
            # list_scan = list(itertools.chain.from_iterable(res))
            write_maldi_ims_chunk_to_imzml1(data, imzml_file, frame_start, frame_stop, mode, exclude_mobility,
                                                profile_bins, encoding, list_scan, rounding = rounding)
            chunk += chunk_size
        else:
            chunk_list = []
            for i, j in zip(frames[chunk:-1], frames[chunk + 1:]):
                chunk_list.append((int(i), int(j)))
            chunk_list.append((j, data.frames.shape[0] + 1))
            logging.info(get_timestamp() + ':' + 'Parsing and writing Frame ' + ':' + str(chunk_list[0][0]) + '...')
            for frame_start, frame_stop in chunk_list:
                write_maldi_ims_chunk_to_imzml(data, imzml_file, frame_start, frame_stop, mode, exclude_mobility,
                                               profile_bins, encoding)
    logging.info(get_timestamp() + ':' + 'Finished writing to .mzML file ' + os.path.join(outdir, outfile) + '...')
