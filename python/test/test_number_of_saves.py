import unittest
import alsvinn
import alsvinn.config
import netCDF4
import subprocess
import os



class TestTextCache(unittest.TestCase):

    def test_cache(self):
        endTime = 1.3
        pythonname = "delete_number_of_saves_sodshocktube.py"
        xmlname = "delete_number_of_saves_sodshocktube.xml"
        output_basename = "delete_number_of_saves_sodshocktube"
        number_of_saves = 3
        xml_content = f"""
<config>
<fvm>
  <name>
    sodtube
  </name>
  <platform>cpu</platform>
  <grid>
    <lowerCorner>-5 0 0</lowerCorner>
    <upperCorner>5 0 0</upperCorner>
    <dimension>16 1 1</dimension>
  </grid>
  <boundary>neumann</boundary>
  <flux>hll3</flux>
  <endTime>{endTime}</endTime>
  <equation>euler1</equation>
  <reconstruction>none</reconstruction>
  <cfl>0.45</cfl>
  <integrator>rungekutta2</integrator>
  <equationParameters>
      <gamma>1.4</gamma>
  </equationParameters>
  <initialData>
    <python>{pythonname}</python>
     <parameters>
      <parameter>
        <name>X</name>
        <length>1</length>
        <value>0</value>
      </parameter>
    </parameters>
  </initialData>
    <diffusion>
        <name>none</name>
        <reconstruction>none</reconstruction>
    </diffusion>
  <writer>
    <type>netcdf</type>
    <basename>{output_basename}</basename>
    <numberOfSaves>{number_of_saves}</numberOfSaves>
  </writer>
</fvm>
<uq>
  <samples>1</samples>
  <generator>auto</generator>
  <parameters>
    <parameter>
      <name>X</name>
      <length>1</length>
      <type>uniform</type>
    </parameter>
  </parameters>
  <stats>
     <stat>
      <name>meanvar</name>
      <numberOfSaves>{number_of_saves}</numberOfSaves>
      <writer>
      <type>netcdf</type>
      <basename>{output_basename}</basename>
      </writer>
    </stat>
  </stats>
</uq>
</config>
        """

        with open(xmlname, 'w') as f:
            f.write(xml_content)

        python_content = f"""
# THIS SHOULD BE CACHED {number_of_saves}
if x < 0.0 + X:
    rho = 1.
    ux = 0.
    p = 1.0
else:
    rho = .125
    ux = 0.
    p = 0.1
        """

        with open(pythonname, 'w') as f:
            f.write(python_content)

        subprocess.run([
            alsvinn.config.ALSVINNCLI_PATH,
            xmlname],
                        check=True)

        absolute_path_python = os.path.abspath(pythonname)
        absolute_path_xml = os.path.abspath(xmlname)

        absolute_path_python_attr = absolute_path_python.replace("/", "_dash_")
        absolute_path_xml_attr = absolute_path_xml.replace("/", "_dash_")

        python_key = f"alsvinn_report.loadedTextFiles.{absolute_path_python_attr}"
        xml_key = f"alsvinn_report.loadedTextFiles.{absolute_path_xml_attr}"

        for timestep in range(number_of_saves+1):
            with netCDF4.Dataset(f"{output_basename}_{timestep}.nc") as f:

                wanted_time = endTime * timestep * 1.0 / (number_of_saves)
                time = float(f.variables['time'][0])

                self.assertEqual(wanted_time, time)
                
                python_from_dataset = f.getncattr(python_key)
                self.assertEqual(python_content, python_from_dataset)


                xml_from_dataset = f.getncattr(xml_key)
                self.assertEqual(xml_content, xml_from_dataset)


                
        subprocess.run([
            alsvinn.config.ALSUQCLI_PATH,
            xmlname],
                       check=True)

        for timestep in range(3):
            with netCDF4.Dataset(f"{output_basename}_{timestep}.nc") as f:
                
                wanted_time = endTime * timestep * 1.0 / (number_of_saves)
                time = float(f.variables['time'][0])

                self.assertEqual(wanted_time, time)

                python_from_dataset = f.getncattr(python_key)
                self.assertEqual(python_content, python_from_dataset)

                xml_from_dataset = f.getncattr(xml_key)
                self.assertEqual(xml_content, xml_from_dataset)

            for stat in ['mean', 'variance']:
                with netCDF4.Dataset(f"{output_basename}_{stat}_{timestep}.nc") as f:
                    
                    wanted_time = endTime * timestep * 1.0 / (number_of_saves)
                    time = float(f.variables['time'][0])

                    self.assertEqual(wanted_time, time)

                    python_from_dataset = f.getncattr(python_key)
                    self.assertEqual(python_content, python_from_dataset)
                    
                    xml_from_dataset = f.getncattr(xml_key)
                    self.assertEqual(xml_content, xml_from_dataset)



if __name__ == '__main__':
    unittest.main()
