


## POSTSHOT?

https://www.jawset.com/


## NERFSTUDIO

 sudo mount -t ntfs /dev/nvme0n1p3 /media/mschuety/Tails
conda activate nerfstudio


https://docs.nerf.studio/quickstart/custom_dataset.html
https://docs.nerf.studio/nerfology/methods/splat.html




<!-- ns-process-data images --data /media/mschuety/Tails/resources/images/australia_gas_station/images --output-dir ~/dev/temp/nstest2/nsdata
ns-train splatfacto --data ~/dev/temp/nstest2/nsdata
ns-export gaussian-splat --load-config outputs/nsdata/splatfacto/2024-08-16_130305/config.yml --output-dir ~/resources/splats/australia_gas_station

ns-process-data images --data /media/mschuety/Tails/resources/images/photogrammetrie_paris_bridge/images --output-dir /media/mschuety/Tails/resources/images/photogrammetrie_paris_bridge/ns
ns-train splatfacto --data /media/mschuety/Tails/resources/images/photogrammetrie_paris_bridge/ns
ns-export gaussian-splat --load-config outputs/ns/splatfacto/2024-08-16_141651/config.yml --output-dir /media/mschuety/Sonic/resources/gaussian_splats/paris_bridge -->


// convert images to nerfstudio format
datasetName=alley
imageDir=/media/mschuety/Tails/resources/images/${datasetName}/
nsdataDir=/media/mschuety/Tails/resources/nsdata/${datasetName}/
nssplatDir=/media/mschuety/Tails/resources/nssplat/${datasetName}/
splatsDir=/media/mschuety/Tails/resources/splats/${datasetName}/
ns-process-data images --data ${imageDir} --output-dir ${nsdataDir} 

// nerfstudio image to nerfstudio splat
datasetName=wisteria
imageDir=/media/mschuety/Tails/resources/images/${datasetName}/
nsdataDir=/media/mschuety/Tails/resources/nsdata/${datasetName}/
nssplatDir=/media/mschuety/Tails/resources/nssplat/${datasetName}/
splatsDir=/media/mschuety/Tails/resources/splats/${datasetName}/
ns-train splatfacto --data ${nsdataDir} --output-dir ${nssplatDir} --pipeline.model.use-scale-regularization True

// export splat model to ply
exportConfigPath="$(find ${nssplatDir} | grep config.yml)"
ns-export gaussian-splat --load-config ${exportConfigPath} --output-dir /media/mschuety/Tails/resources/splats/${datasetName}


auto image downscale factor of 4                                                 nerfstudio_dataparser.py:484