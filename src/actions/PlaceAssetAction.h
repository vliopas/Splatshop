#pragma once

#include "InputAction.h"

struct PlaceAssetUndoAction : public Action{

	shared_ptr<SceneNode> asset;
	shared_ptr<SceneNode> placedItem;
	mat4 transform;

	~PlaceAssetUndoAction(){
		
	}

	void undo(){
		auto editor = SplatEditor::instance;

		editor->scene.world->remove(placedItem.get());
		// TODO: We should probably remove the placedItem and then recreate it from the actual asset on redo?
	}

	void redo(){
		auto editor = SplatEditor::instance;

		editor->scene.world->children.push_back(placedItem);
		editor->setSelectedNode(placedItem.get());
	}

	int64_t byteSize(){
		return 0; // Not enough to matter.
	}

};

struct PlaceAssetAction : public InputAction{

	shared_ptr<SceneNode> placingItem = nullptr;

	void start(){
		auto editor = SplatEditor::instance;

		editor->scene.placingContainer->children.push_back(placingItem);
	}

	void update(){

		auto editor = SplatEditor::instance;
		auto& settings = editor->settings;

		static bool wasLeftDown = false;
		bool isLeftDown = Runtime::mouseEvents.isLeftDown;
		bool leftReleased = wasLeftDown && !isLeftDown;
		bool leftPressed = wasLeftDown == false && isLeftDown == true;
		wasLeftDown = isLeftDown;


		onTypeMatch<SNSplats>(placingItem, [&](shared_ptr<SNSplats> node){
			node->dmng.data.writeDepth = false;

			vec3 pos = editor->deviceState.hovered_pos;
			if(editor->deviceState.hovered_depth == Infinity){
				pos = {0.0f, 0.0f, 0.0f};
			}else{
				pos = editor->deviceState.hovered_pos;
			}

			editor->scene.placingContainer->transform = translate(editor->deviceState.hovered_pos);

			if(leftPressed){

				// Place a clone into the scene
				shared_ptr<SNSplats> newNode = editor->clone(node.get());
				newNode->hidden = false;
				newNode->dmng.data.writeDepth = true;
				editor->scene.world->children.push_back(newNode);

				newNode->transform = translate(editor->deviceState.hovered_pos);

				editor->setSelectedNode(newNode.get());

				// register an undoable action
				shared_ptr<PlaceAssetUndoAction> action = make_shared<PlaceAssetUndoAction>();
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
	}

	void makeToolbarSettings(){
		
	}

};