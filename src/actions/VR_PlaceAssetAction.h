#pragma once

#include "InputAction.h"

struct VRPlaceAssetUndoAction : public Action{

	shared_ptr<SceneNode> asset;
	shared_ptr<SceneNode> placedItem;
	mat4 transform;

	~VRPlaceAssetUndoAction(){
		
	}

	void undo(){
		auto editor = SplatEditor::instance;

		editor->scene.world->remove(placedItem.get());
		// placedItem = nullptr;
	}

	void redo(){
		auto editor = SplatEditor::instance;

		editor->scene.world->children.push_back(placedItem);
		editor->setSelectedNode(placedItem.get());
	}

	int64_t byteSize(){
		return 0; // A bit tricky. May or may not free up placedItem.
	}

};

struct VR_PlaceAssetAction : public InputAction{

	shared_ptr<SceneNode> placingItem = nullptr;

	void start(){
		auto editor = SplatEditor::instance;

		// editor->scene.world->children.push_back(placingItem);
		editor->scene.placingContainer->children.push_back(placingItem);
		// editor->scene.vr->children.push_back(placingItem);

	}

	void update(){

		auto editor = SplatEditor::instance;
		auto& settings = editor->settings;

		auto ovr = editor->ovr;

		if(!ovr->isActive()) return;

		Controller right = ovr->getRightController();

		onTypeMatch<SNSplats>(placingItem, [&](shared_ptr<SNSplats> node){
			node->dmng.data.writeDepth = false;

			// vec3 pos = ovr->flip * right.transform * vec4(0.0f, 0.0f, 0.0f, 1.0f);
			// mat4 world = editor->scene.world->transform;
			// editor->scene.placingContainer->transform = translate(editor->deviceState.hovered_pos);
			// editor->scene.placingContainer->transform = inverse(world) * translate(editor->deviceState.hovered_pos);
			
			vec3 pos = {0.0f, 0.0f, 0.0f};
			{
			
				if(right.valid){
					pos = vec3(ovr->flip * right.transform * vec4(0.0f, 0.0f, 0.0f, 1.0f));
				}

				vec3 p0 = vec3(editor->scene.world->transform * vec4{0.0f, 0.0f, 0.0f, 1.0f});
				vec3 p1 = vec3(editor->scene.world->transform * vec4{1.0f, 0.0f, 0.0f, 1.0f});
				float scale = length(p1 - p0);

				mat4 transform = glm::translate(pos) * glm::scale(vec3{scale, scale, scale});

				placingItem->transform = transform;
			}

			if(right.triggered()){

				// Place a clone into the scene
				shared_ptr<SNSplats> newNode = editor->clone(node.get());
				newNode->hidden = false;
				editor->scene.world->children.push_back(newNode);

				// newNode->transform = translate(pos);
				newNode->transform = inverse(editor->scene.world->transform) * placingItem->transform;
				

				editor->setSelectedNode(newNode.get());

				// register an undoable action
				shared_ptr<VRPlaceAssetUndoAction> action = make_shared<VRPlaceAssetUndoAction>();
				action->asset = placingItem;
				action->placedItem = newNode;
				action->transform = newNode->transform;
				editor->addAction(action);
			}
		});
	}

	void stop(){
		// endUndoable();
		auto editor = SplatEditor::instance;

		editor->scene.placingContainer->children.clear();
		// editor->scene.world->remove(placingItem.get());
	}

	void makeToolbarSettings(){
		
	}

};