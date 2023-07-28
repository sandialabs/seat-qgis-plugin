<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis minScale="1e+08" styleCategories="AllStyleCategories" version="3.28.2-Firenze" hasScaleBasedVisibilityFlag="0" maxScale="0">
  <flags>
    <Identifiable>1</Identifiable>
    <Removable>1</Removable>
    <Searchable>1</Searchable>
    <Private>0</Private>
  </flags>
  <temporal enabled="0" fetchMode="0" mode="0">
    <fixedRange>
      <start></start>
      <end></end>
    </fixedRange>
  </temporal>
  <elevation enabled="0" zoffset="0" zscale="1" band="1" symbology="Line">
    <data-defined-properties>
      <Option type="Map">
        <Option type="QString" value="" name="name"/>
        <Option name="properties"/>
        <Option type="QString" value="collection" name="type"/>
      </Option>
    </data-defined-properties>
    <profileLineSymbol>
      <symbol frame_rate="10" force_rhr="0" alpha="1" is_animated="0" type="line" clip_to_extent="1" name="">
        <data_defined_properties>
          <Option type="Map">
            <Option type="QString" value="" name="name"/>
            <Option name="properties"/>
            <Option type="QString" value="collection" name="type"/>
          </Option>
        </data_defined_properties>
        <layer enabled="1" pass="0" class="SimpleLine" locked="0">
          <Option type="Map">
            <Option type="QString" value="0" name="align_dash_pattern"/>
            <Option type="QString" value="square" name="capstyle"/>
            <Option type="QString" value="5;2" name="customdash"/>
            <Option type="QString" value="3x:0,0,0,0,0,0" name="customdash_map_unit_scale"/>
            <Option type="QString" value="MM" name="customdash_unit"/>
            <Option type="QString" value="0" name="dash_pattern_offset"/>
            <Option type="QString" value="3x:0,0,0,0,0,0" name="dash_pattern_offset_map_unit_scale"/>
            <Option type="QString" value="MM" name="dash_pattern_offset_unit"/>
            <Option type="QString" value="0" name="draw_inside_polygon"/>
            <Option type="QString" value="bevel" name="joinstyle"/>
            <Option type="QString" value="114,155,111,255" name="line_color"/>
            <Option type="QString" value="solid" name="line_style"/>
            <Option type="QString" value="0.6" name="line_width"/>
            <Option type="QString" value="MM" name="line_width_unit"/>
            <Option type="QString" value="0" name="offset"/>
            <Option type="QString" value="3x:0,0,0,0,0,0" name="offset_map_unit_scale"/>
            <Option type="QString" value="MM" name="offset_unit"/>
            <Option type="QString" value="0" name="ring_filter"/>
            <Option type="QString" value="0" name="trim_distance_end"/>
            <Option type="QString" value="3x:0,0,0,0,0,0" name="trim_distance_end_map_unit_scale"/>
            <Option type="QString" value="MM" name="trim_distance_end_unit"/>
            <Option type="QString" value="0" name="trim_distance_start"/>
            <Option type="QString" value="3x:0,0,0,0,0,0" name="trim_distance_start_map_unit_scale"/>
            <Option type="QString" value="MM" name="trim_distance_start_unit"/>
            <Option type="QString" value="0" name="tweak_dash_pattern_on_corners"/>
            <Option type="QString" value="0" name="use_custom_dash"/>
            <Option type="QString" value="3x:0,0,0,0,0,0" name="width_map_unit_scale"/>
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option type="QString" value="" name="name"/>
              <Option name="properties"/>
              <Option type="QString" value="collection" name="type"/>
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
    </profileLineSymbol>
    <profileFillSymbol>
      <symbol frame_rate="10" force_rhr="0" alpha="1" is_animated="0" type="fill" clip_to_extent="1" name="">
        <data_defined_properties>
          <Option type="Map">
            <Option type="QString" value="" name="name"/>
            <Option name="properties"/>
            <Option type="QString" value="collection" name="type"/>
          </Option>
        </data_defined_properties>
        <layer enabled="1" pass="0" class="SimpleFill" locked="0">
          <Option type="Map">
            <Option type="QString" value="3x:0,0,0,0,0,0" name="border_width_map_unit_scale"/>
            <Option type="QString" value="114,155,111,255" name="color"/>
            <Option type="QString" value="bevel" name="joinstyle"/>
            <Option type="QString" value="0,0" name="offset"/>
            <Option type="QString" value="3x:0,0,0,0,0,0" name="offset_map_unit_scale"/>
            <Option type="QString" value="MM" name="offset_unit"/>
            <Option type="QString" value="35,35,35,255" name="outline_color"/>
            <Option type="QString" value="no" name="outline_style"/>
            <Option type="QString" value="0.26" name="outline_width"/>
            <Option type="QString" value="MM" name="outline_width_unit"/>
            <Option type="QString" value="solid" name="style"/>
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option type="QString" value="" name="name"/>
              <Option name="properties"/>
              <Option type="QString" value="collection" name="type"/>
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
    </profileFillSymbol>
  </elevation>
  <customproperties>
    <Option type="Map">
      <Option type="bool" value="false" name="WMSBackgroundLayer"/>
      <Option type="bool" value="false" name="WMSPublishDataSourceUrl"/>
      <Option type="int" value="0" name="embeddedWidgets/count"/>
      <Option type="QString" value="Value" name="identify/format"/>
    </Option>
  </customproperties>
  <pipe-data-defined-properties>
    <Option type="Map">
      <Option type="QString" value="" name="name"/>
      <Option name="properties"/>
      <Option type="QString" value="collection" name="type"/>
    </Option>
  </pipe-data-defined-properties>
  <pipe>
    <provider>
      <resampling enabled="false" zoomedInResamplingMethod="nearestNeighbour" zoomedOutResamplingMethod="nearestNeighbour" maxOversampling="2"/>
    </provider>
    <rasterrenderer alphaBand="-1" classificationMax="0.0123687" nodataColor="" band="1" opacity="1" type="singlebandpseudocolor" classificationMin="-3.041115">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>MinMax</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Estimated</statAccuracy>
        <cumulativeCutLower>0.02</cumulativeCutLower>
        <cumulativeCutUpper>0.98</cumulativeCutUpper>
        <stdDevFactor>2</stdDevFactor>
      </minMaxOrigin>
      <rastershader>
        <colorrampshader maximumValue="0.0123687" colorRampType="INTERPOLATED" minimumValue="-3.041115" labelPrecision="4" classificationMode="1" clip="0">
          <colorramp type="gradient" name="[source]">
            <Option type="Map">
              <Option type="QString" value="68,1,84,255" name="color1"/>
              <Option type="QString" value="253,231,37,255" name="color2"/>
              <Option type="QString" value="ccw" name="direction"/>
              <Option type="QString" value="0" name="discrete"/>
              <Option type="QString" value="gradient" name="rampType"/>
              <Option type="QString" value="rgb" name="spec"/>
              <Option type="QString" value="0.0196078;70,8,92,255;rgb;ccw:0.0392157;71,16,99,255;rgb;ccw:0.0588235;72,23,105,255;rgb;ccw:0.0784314;72,29,111,255;rgb;ccw:0.0980392;72,36,117,255;rgb;ccw:0.117647;71,42,122,255;rgb;ccw:0.137255;70,48,126,255;rgb;ccw:0.156863;69,55,129,255;rgb;ccw:0.176471;67,61,132,255;rgb;ccw:0.196078;65,66,135,255;rgb;ccw:0.215686;63,72,137,255;rgb;ccw:0.235294;61,78,138,255;rgb;ccw:0.254902;58,83,139,255;rgb;ccw:0.27451;56,89,140,255;rgb;ccw:0.294118;53,94,141,255;rgb;ccw:0.313725;51,99,141,255;rgb;ccw:0.333333;49,104,142,255;rgb;ccw:0.352941;46,109,142,255;rgb;ccw:0.372549;44,113,142,255;rgb;ccw:0.392157;42,118,142,255;rgb;ccw:0.411765;41,123,142,255;rgb;ccw:0.431373;39,128,142,255;rgb;ccw:0.45098;37,132,142,255;rgb;ccw:0.470588;35,137,142,255;rgb;ccw:0.490196;33,142,141,255;rgb;ccw:0.509804;32,146,140,255;rgb;ccw:0.529412;31,151,139,255;rgb;ccw:0.54902;30,156,137,255;rgb;ccw:0.568627;31,161,136,255;rgb;ccw:0.588235;33,165,133,255;rgb;ccw:0.607843;36,170,131,255;rgb;ccw:0.627451;40,174,128,255;rgb;ccw:0.647059;46,179,124,255;rgb;ccw:0.666667;53,183,121,255;rgb;ccw:0.686275;61,188,116,255;rgb;ccw:0.705882;70,192,111,255;rgb;ccw:0.72549;80,196,106,255;rgb;ccw:0.745098;90,200,100,255;rgb;ccw:0.764706;101,203,94,255;rgb;ccw:0.784314;112,207,87,255;rgb;ccw:0.803922;124,210,80,255;rgb;ccw:0.823529;137,213,72,255;rgb;ccw:0.843137;149,216,64,255;rgb;ccw:0.862745;162,218,55,255;rgb;ccw:0.882353;176,221,47,255;rgb;ccw:0.901961;189,223,38,255;rgb;ccw:0.921569;202,225,31,255;rgb;ccw:0.941176;216,226,25,255;rgb;ccw:0.960784;229,228,25,255;rgb;ccw:0.980392;241,229,29,255;rgb;ccw" name="stops"/>
            </Option>
          </colorramp>
          <item label="-3.0411" alpha="255" color="#440154" value="-3.0411150455475"/>
          <item label="-2.9812" alpha="255" color="#46085c" value="-2.981242947135008"/>
          <item label="-2.9214" alpha="255" color="#471063" value="-2.921370543374142"/>
          <item label="-2.8615" alpha="255" color="#481769" value="-2.86149844496165"/>
          <item label="-2.8016" alpha="255" color="#481d6f" value="-2.801626041200784"/>
          <item label="-2.7418" alpha="255" color="#482475" value="-2.741753942788292"/>
          <item label="-2.6819" alpha="255" color="#472a7a" value="-2.681881844375799"/>
          <item label="-2.6220" alpha="255" color="#46307e" value="-2.62200913526656"/>
          <item label="-2.5621" alpha="255" color="#453781" value="-2.56213642615732"/>
          <item label="-2.5023" alpha="255" color="#433d84" value="-2.50226371704808"/>
          <item label="-2.4424" alpha="255" color="#414287" value="-2.442394061422578"/>
          <item label="-2.3825" alpha="255" color="#3f4889" value="-2.382521352313338"/>
          <item label="-2.3226" alpha="255" color="#3d4e8a" value="-2.322648643204099"/>
          <item label="-2.2628" alpha="255" color="#3a538b" value="-2.262775934094859"/>
          <item label="-2.2029" alpha="255" color="#38598c" value="-2.20290322498562"/>
          <item label="-2.1430" alpha="255" color="#355e8d" value="-2.14303051587638"/>
          <item label="-2.0832" alpha="255" color="#33638d" value="-2.083160860250877"/>
          <item label="-2.0233" alpha="255" color="#31688e" value="-2.023288151141638"/>
          <item label="-1.9634" alpha="255" color="#2e6d8e" value="-1.963415442032398"/>
          <item label="-1.9035" alpha="255" color="#2c718e" value="-1.903542732923158"/>
          <item label="-1.8437" alpha="255" color="#2a768e" value="-1.843670023813919"/>
          <item label="-1.7838" alpha="255" color="#297b8e" value="-1.78379731470468"/>
          <item label="-1.7239" alpha="255" color="#27808e" value="-1.72392460559544"/>
          <item label="-1.6641" alpha="255" color="#25848e" value="-1.664054949969937"/>
          <item label="-1.6042" alpha="255" color="#23898e" value="-1.604182240860697"/>
          <item label="-1.5443" alpha="255" color="#218e8d" value="-1.544309531751458"/>
          <item label="-1.4844" alpha="255" color="#20928c" value="-1.484436822642218"/>
          <item label="-1.4246" alpha="255" color="#1f978b" value="-1.424564113532979"/>
          <item label="-1.3647" alpha="255" color="#1e9c89" value="-1.36469140442374"/>
          <item label="-1.3048" alpha="255" color="#1fa188" value="-1.304821748798236"/>
          <item label="-1.2449" alpha="255" color="#21a585" value="-1.244949039688997"/>
          <item label="-1.1851" alpha="255" color="#24aa83" value="-1.185076330579757"/>
          <item label="-1.1252" alpha="255" color="#28ae80" value="-1.125203621470518"/>
          <item label="-1.0653" alpha="255" color="#2eb37c" value="-1.065330912361278"/>
          <item label="-1.0055" alpha="255" color="#35b779" value="-1.005458203252038"/>
          <item label="-0.9456" alpha="255" color="#3dbc74" value="-0.945585494142799"/>
          <item label="-0.8857" alpha="255" color="#46c06f" value="-0.885715838517296"/>
          <item label="-0.8258" alpha="255" color="#50c46a" value="-0.825843129408057"/>
          <item label="-0.7660" alpha="255" color="#5ac864" value="-0.765970420298817"/>
          <item label="-0.7061" alpha="255" color="#65cb5e" value="-0.706097711189578"/>
          <item label="-0.6462" alpha="255" color="#70cf57" value="-0.646225002080338"/>
          <item label="-0.5864" alpha="255" color="#7cd250" value="-0.586352292971098"/>
          <item label="-0.5265" alpha="255" color="#89d548" value="-0.526482637345596"/>
          <item label="-0.4666" alpha="255" color="#95d840" value="-0.466609928236356"/>
          <item label="-0.4067" alpha="255" color="#a2da37" value="-0.406737219127117"/>
          <item label="-0.3469" alpha="255" color="#b0dd2f" value="-0.346864510017876"/>
          <item label="-0.2870" alpha="255" color="#bddf26" value="-0.286991800908637"/>
          <item label="-0.2271" alpha="255" color="#cae11f" value="-0.227119091799398"/>
          <item label="-0.1672" alpha="255" color="#d8e219" value="-0.167249436173895"/>
          <item label="-0.1074" alpha="255" color="#e5e419" value="-0.107376727064656"/>
          <item label="-0.0475" alpha="255" color="#f1e51d" value="-0.0475040179554154"/>
          <item label="0.0124" alpha="255" color="#fde725" value="0.0123686911538239"/>
          <rampLegendSettings minimumLabel="" orientation="2" direction="0" maximumLabel="" prefix="" suffix="" useContinuousLegend="1">
            <numericFormat id="basic">
              <Option type="Map">
                <Option type="invalid" name="decimal_separator"/>
                <Option type="int" value="6" name="decimals"/>
                <Option type="int" value="0" name="rounding_type"/>
                <Option type="bool" value="false" name="show_plus"/>
                <Option type="bool" value="true" name="show_thousand_separator"/>
                <Option type="bool" value="false" name="show_trailing_zeros"/>
                <Option type="invalid" name="thousand_separator"/>
              </Option>
            </numericFormat>
          </rampLegendSettings>
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
    <brightnesscontrast gamma="1" brightness="0" contrast="0"/>
    <huesaturation colorizeOn="0" invertColors="0" colorizeRed="255" colorizeStrength="100" colorizeGreen="128" saturation="0" grayscaleMode="0" colorizeBlue="128"/>
    <rasterresampler maxOversampling="2"/>
    <resamplingStage>resamplingFilter</resamplingStage>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
