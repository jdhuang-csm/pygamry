import time
import comtypes.client as client


dtaq_header_info = {
    'GamryDtaqOcv': {
        'preceding': 'CURVE\tTABLE',
        'columns': ['Pt', 'Time', 'Vf', 'Vm', 'Ach'],
        'units': ['#', 's', 'V vs. Ref.', 'V', 'V'],
        'cook_columns': ['INDEX', 'Time', 'Vf', 'Vm', 'Ach'],
        # 'kst_map': {'Time': 'Time', 'Vf': 'OCV'},
        'kst_columns': ['Time', 'Vf']
    },
    'GamryDtaqCpiv': {
        'preceding': 'CURVE\tTABLE',
        'columns': ['Pt', 'Time', 'Vf', 'Im', 'Vu', 'Sig', 'Ach', 'IERange'],
        'units': ['#', 's', 'V vs. Ref.', 'A', 'V', 'V', 'V', '#'],
        'cook_columns': ['INDEX', 'Time', 'Vf', 'Im', 'Vu', 'Vsig', 'Ach', 'IERange'],
        # 'kst_map': {'Time': 'Time', 'Vf': 'V', 'Im': 'I'},
        'kst_columns': ['Time', 'Vf', 'Im']
    },
    'GamryDtaqCiiv': {
        'preceding': 'CURVE\tTABLE',
        'columns': ['Pt', 'Time', 'Vf', 'Im', 'Vu', 'Sig', 'Ach', 'IERange'],
        'units': ['#', 's', 'V vs. Ref.', 'A', 'V', 'V', 'V', '#'],
        'cook_columns': ['INDEX', 'Time', 'Vf', 'Im', 'Vu', 'Vsig', 'Ach', 'IERange'],
        # 'kst_map': {'Time': 'Time', 'Vf': 'V', 'Im': 'I'},
        'kst_columns': ['Time', 'Vf', 'Im']
    },
    'GamryDtaqPwr': {
        'preceding': 'CURVE\tTABLE',
        'columns': ['Pt', 'Time', 'Vf', 'Im', 'Vu', 'Pwr', 'Sig', 'Ach', 'IERange', 'ImExpected'],
        'units': ['#', 's', 'V vs. Ref.', 'A', 'V', 'W', 'V', 'V', '#', 'A'],
        'cook_columns': ['INDEX', 'Time', 'Vf', 'Im', 'Vu', 'Pwr', 'Vsig', 'Ach', 'IERange'],
        # 'kst_map': {'Vf': 'V', 'Im': 'I', 'Pwr': 'P'},
        'kst_columns': ['Vf', 'Im', 'Pwr']
    },
    'GamryReadZ': {
        'preceding': 'ZCURVE\tTABLE',
        'columns': ['Pt', 'Time', 'Freq', 'Zreal', 'Zimag', 'Zsig', 'Zmod', 'Zphz', 'Idc', 'Vdc', 'IERange'],
        'units': ['#', 's', 'Hz', 'ohm', 'ohm', 'V', 'ohm', '°', 'A', 'V', '#'],
        'cook_columns': ['INDEX'],
        # 'kst_map': {'Freq': 'f', 'Zreal': "Z'", 'Zimag': "Z''", 'Zmod': '|Z|', 'Zphz': 'Phase'},
        'kst_columns': ['Freq', 'Zreal', 'Zimag', 'Zmod', 'Zphz']
    },
    'GamryDtaqChrono': {
        'preceding': 'CURVE\tTABLE',
        'columns': ['Pt', 'Time', 'Vf', 'Im', 'Vu', 'Sig', 'Ach', 'IERange'],
        'units': ['#', 's', 'V vs. Ref.', 'A', 'V', 'V', 'V', '#'],
        'cook_columns': ['INDEX', 'Time', 'Vf', 'Im', 'Vu', 'Vsig', 'Ach', 'IERange'],
        # 'kst_map': {'Time': 'Time', 'Vf': 'V', 'Im': 'I'},
        'kst_columns': ['Time', 'Vf', 'Im']
    }
}
# Cook column lookup
# -------------------
dtaq_cook_columns = {
    'GamryDtaqChrono': [
        'Time',
        'Vf',
        'Vu',
        'Im',
        'Q',
        'Vsig',
        'Ach',
        'IERange',
        'Overload',
        'StopTest'
    ],
    'GamryDtaqOcv': [
        'Time',
        'Vf',
        'Vm',
        'Vsig',
        'Ach',
        'Overload',
        'StopTest',
        # Undocumented columns
        'Ignore',
        'Ignore',
        'Ignore'
    ],
    'GamryDtaqCpiv': [
        'Time',
        'Vf',
        'Vu',
        'Im',
        'Vsig',
        'Ach',
        'IERange',
        'Overload',
        'StopTest'
    ],
    'GamryDtaqCiiv': [
        'Time',
        'Vf',
        'Vu',
        'Im',
        'Vsig',
        'Ach',
        'IERange',
        'Overload',
        'StopTest'
    ],
    'GamryDtaqPwr': [
        'Time',
        'Vf',
        'Vu',
        'Im',
        'Pwr',
        'R',
        'Vsig',
        'Ach',
        'Temp',
        'IERange',
        'Overload',
        'StopTest',
        'StopTest2'
    ],
    'GamryDtaqEis': [
        'I',
        'V'
    ],
    'GamryReadZ': [
        'I',
        'V'
    ],
}
GamryCOM = client.GetModule(['{BD962F0D-A990-4823-9CF5-284D1CDD9C6D}', 1, 0])

# ================================
# GamryCOM mapping
# ================================
# TODO: this isn't really necessary, find all usages and remove
def get_gc_to_str_map(string_list):
    return {getattr(GamryCOM, string): string for string in string_list}

def get_str_to_gc_map(string_list):
    return {string: getattr(GamryCOM, string) for string in string_list}

gc_string_dict = {
    'CtrlMode': ['GstatMode', 'PstatMode'],
}


# TODO: by default, get first pstat of any type
def get_pstat(family='Interface', retry=5, device_index=0):
    """
    Get potentiostat
    :param str family: potentiostat family. Options: 'Interface', 'Reference'
    :return:
    """
    pstat = None
    iteration = 0
    while pstat is None:
        try:
            devices = client.CreateObject('GamryCOM.GamryDeviceList')
            print(devices.EnumSections())

            if family.lower() == 'interface':
                obj_string = 'GamryCOM.GamryPC6Pstat'
            elif family.lower() == 'reference':
                obj_string = 'GamryCOM.GamryPC5Pstat'
            else:
                raise ValueError(f'Invalid family argument {family}')

            pstat = client.CreateObject(obj_string)
            pstat.Init(devices.EnumSections()[device_index])
            return pstat
        except Exception as err:
            pstat = None
            iteration += 1

            if iteration == retry:
                print('Could not find an available potentiostat')
                raise (err)
            else:
                print('No pstat available. Retrying in 1 s')
                time.sleep(1)
