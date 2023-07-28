<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis minScale="1e+08" styleCategories="AllStyleCategories" hasScaleBasedVisibilityFlag="0" maxScale="0" version="3.22.10-Białowieża">
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
  <customproperties>
    <Option type="Map">
      <Option name="WMSBackgroundLayer" type="bool" value="false"/>
      <Option name="WMSPublishDataSourceUrl" type="bool" value="false"/>
      <Option name="embeddedWidgets/count" type="int" value="0"/>
      <Option name="identify/format" type="QString" value="Value"/>
    </Option>
  </customproperties>
  <pipe-data-defined-properties>
    <Option type="Map">
      <Option name="name" type="QString" value=""/>
      <Option name="properties"/>
      <Option name="type" type="QString" value="collection"/>
    </Option>
  </pipe-data-defined-properties>
  <pipe>
    <provider>
      <resampling enabled="false" zoomedInResamplingMethod="nearestNeighbour" maxOversampling="2" zoomedOutResamplingMethod="nearestNeighbour"/>
    </provider>
    <rasterrenderer band="1" alphaBand="-1" nodataColor="" opacity="1" type="paletted">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>None</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Estimated</statAccuracy>
        <cumulativeCutLower>0.02</cumulativeCutLower>
        <cumulativeCutUpper>0.98</cumulativeCutUpper>
        <stdDevFactor>2</stdDevFactor>
      </minMaxOrigin>
      <colorPalette>
        <paletteEntry label="Increased Deposition" color="#ca0020" value="-2" alpha="255"/>
        <paletteEntry label="Reduced Deposition" color="#f4a582" value="-1" alpha="255"/>
        <paletteEntry label="No Change" color="#f7f7f7" value="0" alpha="255"/>
        <paletteEntry label="Reduced Erosion" color="#92c5de" value="1" alpha="255"/>
        <paletteEntry label="Increased Erosion" color="#0571b0" value="2" alpha="255"/>
      </colorPalette>
      <colorramp name="[source]" type="gradient">
        <Option type="Map">
          <Option name="color1" type="QString" value="202,0,32,255"/>
          <Option name="color2" type="QString" value="5,113,176,255"/>
          <Option name="discrete" type="QString" value="0"/>
          <Option name="rampType" type="QString" value="gradient"/>
          <Option name="stops" type="QString" value="0.25;244,165,130,255:0.5;247,247,247,255:0.75;146,197,222,255"/>
        </Option>
        <prop v="202,0,32,255" k="color1"/>
        <prop v="5,113,176,255" k="color2"/>
        <prop v="0" k="discrete"/>
        <prop v="gradient" k="rampType"/>
        <prop v="0.25;244,165,130,255:0.5;247,247,247,255:0.75;146,197,222,255" k="stops"/>
      </colorramp>
    </rasterrenderer>
    <brightnesscontrast contrast="0" brightness="0" gamma="1"/>
    <huesaturation colorizeBlue="128" saturation="0" colorizeGreen="128" colorizeOn="0" colorizeRed="255" grayscaleMode="0" colorizeStrength="100" invertColors="0"/>
    <rasterresampler maxOversampling="2"/>
    <resamplingStage>resamplingFilter</resamplingStage>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
